#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bagごとにトラックを構成し、各トラックに
- 連続フレーム間のエピポーラ誤差 (i→i+1)
- 0→k のエピポーラ誤差 (0→2,0→3,..., start_id=0 のトラックのみ)

を保存するだけのシンプルなスクリプト。

マッチャー:
  - JamMa (時系列 prev_data あり)
  - SuperPoint + LightGlue
  - legacy-JamMa (最近傍リンク)

出力:
  手法ごとに JSON を 1 つずつ書き出す。
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Any

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pytorch_lightning as pl

# --- JamMa imports (project-local) ---
from src.utils.dataset import read_megadepth_color
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler

# --- LightGlue imports ---
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# --- Epipolar metrics (JamMa実装) ---
from src.utils.metrics import (
    symmetric_epipolar_distance,
)

from thop import profile


# ============================================================
# 1) Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- JamMa config ---
    parser.add_argument('--data_cfg_path', type=str,
                        default="configs/data/megadepth_test_1500.py")
    parser.add_argument('--main_cfg_path', type=str,
                        default="configs/jamma/outdoor/test.py")
    parser.add_argument('--ckpt_path', type=str, default="official")
    parser.add_argument('--dump_dir', type=str, default="dump/jamma_outdoor")
    parser.add_argument('--profiler_name', type=str, default="inference")

    # --- Dataset defaults (PhotoTourism example) ---
    default_root = Path('/home/ach17765lb/data/phototourism')
    scene = 'sacre_coeur'      # 'reichstag', 'sacre_coeur', 'st_peters_square'
    set_name = 'set_100'

    parser.add_argument('--subset_dir', type=Path,
                        default=default_root / scene / set_name / 'sub_set',
                        help='Contains Nbag_*.txt files (e.g., 5bag_015.txt)')
    parser.add_argument('--dataset_root', type=Path,
                        default=default_root / scene / set_name,
                        help='Root to prepend to image relative paths in bag files')
    parser.add_argument('--calib_dir', type=Path,
                        default=default_root / scene / set_name / 'calibration',
                        help='Directory with calibration_<stem>.h5 per image')
    parser.add_argument('--bag_size', type=int, default=10,
                        help='Number of images in each bag file')
    parser.add_argument('--flip_w2c', action='store_true',
                        help='If calib is world->cam, convert to cam->world internally')

    parser.add_argument('--device', type=str, default='cuda')

    # 手法ごとにファイル名を変えるので、ここは「ベースパス」だけ。
    parser.add_argument(
        '--save_json_base',
        type=Path,
        default=default_root / 'tracks_epipolar'
    )

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


# ============================================================
# 2) Calibration utilities
# ============================================================

@dataclasses.dataclass
class CameraParams:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


def _read_cam_from_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, 'r') as f:
        if 'K' in f and 'R' in f and ('T' in f or 't' in f):
            K = np.array(f['K'])
            R = np.array(f['R'])
            T = np.array(f['T'] if 'T' in f else f['t'])
        else:
            K = R = T = None
            for key in f.keys():
                g = f[key]
                if isinstance(g, h5py.Group) and {'K', 'R'}.issubset(g.keys()) and ('T' in g or 't' in g):
                    K = np.array(g['K'])
                    R = np.array(g['R'])
                    T = np.array(g['T'] if 'T' in g else g['t'])
                    break
            if K is None or R is None or T is None:
                raise KeyError(f"K/R/T not found in {h5_path}")
    return K.reshape(3, 3), R.reshape(3, 3), T.reshape(3)


def load_cam_from_dir(calib_dir: Path, img_path: Path, flip_w2c: bool) -> CameraParams:
    h5_path = calib_dir / f"calibration_{img_path.stem}.h5"
    K, R, t = _read_cam_from_h5(h5_path)
    if flip_w2c:
        R, t = R.T, -R.T @ t
    return CameraParams(K, R, t)


def _relative_pose(camA: CameraParams, camB: CameraParams):
    """Relative pose of camB wrt camA."""
    R21 = camB.R @ camA.R.T
    t21 = camB.t - R21 @ camA.t
    return R21, t21


def _compute_symmetric_epi_errors_for_two_cams(
    x0_px: np.ndarray,
    x1_px: np.ndarray,
    cam0: CameraParams,
    cam1: CameraParams,
    device: torch.device,
) -> np.ndarray:
    """
    JamMa と同じ symmetric_epipolar_distance を使ってエピ誤差を計算。
    """
    if x0_px.size == 0:
        return np.zeros((0,), dtype=float)

    # 相対姿勢 cam0->cam1
    R01, t01 = _relative_pose(cam0, cam1)
    t_x = np.array([[0, -t01[2], t01[1]],
                    [t01[2], 0, -t01[0]],
                    [-t01[1], t01[0], 0]], dtype=np.float32)
    E = t_x @ R01

    pts0 = torch.from_numpy(x0_px).float().to(device)
    pts1 = torch.from_numpy(x1_px).float().to(device)
    E_t = torch.from_numpy(E).float().to(device)
    K0_t = torch.from_numpy(cam0.K).float().to(device)
    K1_t = torch.from_numpy(cam1.K).float().to(device)

    errs = symmetric_epipolar_distance(pts0, pts1, E_t, K0_t, K1_t)
    return errs.detach().cpu().numpy().astype(np.float64)


# ============================================================
# 3) Matcher forward per pair
# ============================================================

class SuperPointFlopsWrapper(nn.Module):
    def __init__(self, sp_model):
        super().__init__()
        self.sp = sp_model

    def forward(self, image: torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        feats = self.sp.extract(image)
        return feats["descriptors"]


class LightGlueFlopsWrapper(nn.Module):
    def __init__(self, lg_model):
        super().__init__()
        self.lg = lg_model

    def forward(self, feats0: dict, feats1: dict):
        out = self.lg({"image0": feats0, "image1": feats1})
        return out["matches0"]


def run_jamma_pair(
    device: torch.device,
    imgA: Path,
    imgB: Path,
    prev_data: Any = None,
    jamma: PL_JamMa = None,
):
    """JamMa 1ペア。prev_data を利用して時系列動作させる。"""
    if jamma is None:
        raise ValueError("jamma model must be provided to run_jamma_pair")

    image0, s0, m0, p0, wh0, _ = read_megadepth_color(str(imgA), 832, 8, True)
    image1, s1, m1, p1, wh1,_ = read_megadepth_color(str(imgB), 832, 8, True)

    m0 = F.interpolate(m0[None, None].float(), scale_factor=0.125, mode='nearest')[0].bool()
    m1 = F.interpolate(m1[None, None].float(), scale_factor=0.125, mode='nearest')[0].bool()

    data = dict(
        imagec_0=image0.to(device),
        imagec_1=image1.to(device),
        mask0=m0.to(device),
        mask1=m1.to(device),
        scale0=s0.unsqueeze(0).to(device),
        scale1=s1.unsqueeze(0).to(device),
        prepad_size0=p0.unsqueeze(0).to(device),
        prepad_size1=p1.unsqueeze(0).to(device),
        custom_fine_flex_thr=0.1,
    )
    if prev_data is not None:
        data['prev_data'] = prev_data

    jamma = jamma.to(device).eval()
    with torch.no_grad():
        result, flops, runtime = jamma(data)  # flops: MACs, runtime: ms

    if 'prev_data' in data:
        del data['prev_data']

    mk0 = result['mkpts0_f_origin']
    mk1 = result['mkpts1_f_origin']
    mconf = result.get('mconf_f', None)

    return mk0, mk1, mconf, float(flops), float(runtime), result  # result を次の prev_data に使う


def run_splg_pair(
    device: torch.device,
    imgA: Path,
    imgB: Path,
    algo_res: bool = True,
    prev_data: Any = None,   # インターフェース合わせ。中では使わない。
    sp_model: Any = None,
    lg_model: Any = None,
):
    """SuperPoint + LightGlue で 1 ペア。prev_data は無視。"""
    if sp_model is None or lg_model is None:
        raise ValueError("sp_model と lg_model を渡してください")

    image0 = load_image(str(imgA)).to(device)
    image1 = load_image(str(imgB)).to(device)
    if image0 is None or image1 is None:
        raise FileNotFoundError(f"Failed to read {imgA} or {imgB}")

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None
        import time
        t0 = time.time()

    with torch.no_grad():
        feats0_b = sp_model.extract(image0)
        feats1_b = sp_model.extract(image1)

        matches01_b = lg_model({"image0": feats0_b, "image1": feats1_b})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0_b, feats1_b, matches01_b]]

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        runtime_ms = start_event.elapsed_time(end_event)
    else:
        import time
        runtime_ms = (time.time() - t0) * 1000.0

    matches = matches01['matches']          # (K, 2)
    points0 = feats0['keypoints'][matches[:, 0]]
    points1 = feats1['keypoints'][matches[:, 1]]

    if 'scores' in matches01:
        mconf = matches01['scores']
    else:
        mconf = torch.ones(points0.shape[0], device=device, dtype=torch.float32)

    mk0 = points0.to(device).float()
    mk1 = points1.to(device).float()
    mconf = mconf.to(device).float()

    # FLOPs
    if device.type == "cuda":
        sp_wrap = SuperPointFlopsWrapper(sp_model).to(device).eval()
        lg_wrap = LightGlueFlopsWrapper(lg_model).to(device).eval()
        with torch.no_grad():
            flops_sp0, _ = profile(sp_wrap, inputs=(image0,), verbose=False)
            flops_sp1, _ = profile(sp_wrap, inputs=(image1,), verbose=False)
            flops_lg, _ = profile(lg_wrap, inputs=(feats0_b, feats1_b), verbose=False)
        flops = float(flops_sp0 + flops_sp1 + flops_lg)
    else:
        flops = 0.0

    dummy_prev = {}  # prev_data インターフェース用ダミー

    return mk0, mk1, mconf, flops, runtime_ms, dummy_prev


# ============================================================
# 4) Bag evaluation & tracks
# ============================================================

def read_bag_paths(bag_file: Path) -> List[str]:
    return [ln.strip() for ln in bag_file.read_text(encoding='utf-8').splitlines() if ln.strip()]


def evaluate_bag_and_collect_tracks(
    bag_file: Path,
    bag_size: int,
    dataset_root: Path,
    calib_dir: Path,
    flip_w2c: bool,
    run_pair_fn: Callable[..., Any],
    run_pair_kwargs: dict,
    device: torch.device,
    method_name: str,
) -> dict:
    """
    1 bag について:
      - 全隣接ペア (0-1,1-2,...) をマッチ
      - トラックを構成
      - 各トラックに
          * 連続フレームのエピ誤差
          * 0→k のエピ誤差 (start_id=0 のとき)
        を書き込んで返す。
      - さらに bag 内で「0→k までつながったトラック数」を k ごとに print する。
    """
    rel_paths = read_bag_paths(bag_file)
    if len(rel_paths) != bag_size:
        raise ValueError(f"{bag_file}: expected {bag_size} paths, got {len(rel_paths)}")
    img_paths = [dataset_root / rp for rp in rel_paths]
    cams = [load_cam_from_dir(calib_dir, p, flip_w2c) for p in img_paths]

    tracks: Dict[int, dict] = {}
    next_tid = 0
    num_pairs = bag_size - 1
    num_pairs = bag_size - 1 - 3  # 元コードそのまま踏襲

    # JamMa 時系列用
    prev_data = None

    # linking state
    prev_point_to_tid: Dict[tuple, int] = {}
    prev_points = None
    prev_tids = None

    for i in range(num_pairs):
        imgA, imgB = img_paths[i], img_paths[i + 1]
        print(f"  [{method_name}] Pair {i}: {imgA.name} ↔ {imgB.name}")

        mk0_t, mk1_t, mconf, flops, runtime_ms, prev_out = run_pair_fn(
            device=device,
            imgA=imgA,
            imgB=imgB,
            prev_data=prev_data,
            **run_pair_kwargs,
        )
        prev_data = prev_out

        # confidence でソート → そのまま全部使う（必要なら topk してもOK）
        if mconf is not None and mconf.numel() > 0:
            idx = torch.argsort(mconf, descending=True)
            mk0 = mk0_t[idx].cpu().numpy()
            mk1 = mk1_t[idx].cpu().numpy()
            mconf_np = mconf[idx].cpu().numpy()
        else:
            mk0 = mk0_t.cpu().numpy()
            mk1 = mk1_t.cpu().numpy()
            mconf_np = None

        # ---- track linking ----
        if method_name != "jamma_legacy":
            # JamMa / SP+LG: 座標一致リンクだが、
            # 同じ keyA からは「スコア最大の 1 本だけ」を採用
            curr_point_to_tid: Dict[tuple, int] = {}

            best_matches_per_keyA: Dict[tuple, dict] = {}
            for idx_match, (ptA, ptB) in enumerate(zip(mk0, mk1)):
                keyA = (float(ptA[0]), float(ptA[1]))
                conf_val = float(mconf_np[idx_match]) if mconf_np is not None else 1.0
                if keyA not in best_matches_per_keyA or conf_val > best_matches_per_keyA[keyA]['conf']:
                    best_matches_per_keyA[keyA] = {
                        'ptA': ptA,
                        'ptB': ptB,
                        'conf': conf_val,
                    }

            for keyA, rec in best_matches_per_keyA.items():
                ptA = rec['ptA']
                ptB = rec['ptB']
                conf_val = rec['conf']
                keyB = (float(ptB[0]), float(ptB[1]))

                if keyA in prev_point_to_tid:
                    tid = prev_point_to_tid[keyA]
                    tr = tracks[tid]
                    tr['points'].append([float(ptB[0]), float(ptB[1])])
                    tr['confs'].append(conf_val)
                    tr['end_id'] = i + 1
                else:
                    tid = next_tid
                    next_tid += 1
                    tracks[tid] = {
                        'start_id': i,
                        'end_id': i + 1,
                        'points': [
                            [float(ptA[0]), float(ptA[1])],
                            [float(ptB[0]), float(ptB[1])],
                        ],
                        'confs': [conf_val],
                    }

                curr_point_to_tid[keyB] = tid

            prev_point_to_tid = curr_point_to_tid

        else:
            # legacy: prev_points -> mk0 方向の最近傍リンク
            curr_points_list = []
            curr_tids_list = []

            max_d2 = np.inf
            max_d2 = 50

            if prev_points is None or prev_points.shape[0] == 0:
                for idx_match, (ptA, ptB) in enumerate(zip(mk0, mk1)):
                    conf_val = float(mconf_np[idx_match]) if mconf_np is not None else 1.0
                    tid = next_tid
                    next_tid += 1
                    tracks[tid] = {
                        'start_id': i,
                        'end_id': i + 1,
                        'points': [
                            [float(ptA[0]), float(ptA[1])],
                            [float(ptB[0]), float(ptB[1])],
                        ],
                        'confs': [conf_val],
                    }
                    curr_points_list.append([float(ptB[0]), float(ptB[1])])
                    curr_tids_list.append(tid)

                prev_points = np.array(curr_points_list, dtype=float)
                prev_tids = np.array(curr_tids_list, dtype=int)
            else:
                prev_points_np = prev_points
                prev_tids_np = prev_tids

                N_cur = mk0.shape[0]
                used_current = np.zeros(N_cur, dtype=bool)

                for j, prev_pt in enumerate(prev_points_np):
                    diff = mk0 - prev_pt[None, :]
                    d2 = np.sum(diff * diff, axis=1)
                    k_min = int(np.argmin(d2))
                    if d2[k_min] > max_d2:
                        continue
                    if used_current[k_min]:
                        continue

                    used_current[k_min] = True
                    tid = int(prev_tids_np[j])
                    ptB = mk1[k_min]
                    conf_val = float(mconf_np[k_min]) if mconf_np is not None else 1.0

                    tr = tracks[tid]
                    tr['points'].append([float(ptB[0]), float(ptB[1])])
                    tr['end_id'] = i + 1
                    tr['confs'].append(conf_val)

                    curr_points_list.append([float(ptB[0]), float(ptB[1])])
                    curr_tids_list.append(tid)

                for k in range(N_cur):
                    if used_current[k]:
                        continue
                    ptA = mk0[k]
                    ptB = mk1[k]
                    conf_val = float(mconf_np[k]) if mconf_np is not None else 1.0
                    tid = next_tid
                    next_tid += 1
                    tracks[tid] = {
                        'start_id': i,
                        'end_id': i + 1,
                        'points': [
                            [float(ptA[0]), float(ptA[1])],
                            [float(ptB[0]), float(ptB[1])],
                        ],
                        'confs': [conf_val],
                    }
                    curr_points_list.append([float(ptB[0]), float(ptB[1])])
                    curr_tids_list.append(tid)

                prev_points = np.array(curr_points_list, dtype=float)
                prev_tids = np.array(curr_tids_list, dtype=int)

    # ---- 各トラックにエピ誤差を書き込む ----
    epi_threshold = 1e-4  # 使いたければ使えるように残すが、ここでは保存だけ

    for tid, tr in tracks.items():
        start_id = tr['start_id']
        end_id = tr['end_id']
        pts = np.array(tr['points'], dtype=float)  # shape = [L, 2]
        L = pts.shape[0]

        # 連続フレーム (i→i+1) のエピ誤差
        consecutive_errs = []
        for local_i in range(L - 1):
            frame_i = start_id + local_i
            frame_j = frame_i + 1
            p_i = pts[local_i:local_i + 1]
            p_j = pts[local_i + 1:local_i + 2]
            errs = _compute_symmetric_epi_errors_for_two_cams(
                p_i, p_j, cams[frame_i], cams[frame_j], device=device
            )
            consecutive_errs.append(float(errs[0]))
        tr['epi_errs_consecutive'] = consecutive_errs  # 長さ L-1

        # 0→k のエピ誤差（グローバルフレーム 0 基準）
        epi_0_to_k = []
        if start_id == 0 and L >= 2:
            p0 = pts[0:1]
            for local_k in range(1, L):
                frame_k = local_k  # start_id==0 なので local_k==frameIdx
                pk = pts[local_k:local_k + 1]
                errs = _compute_symmetric_epi_errors_for_two_cams(
                    p0, pk, cams[0], cams[frame_k], device=device
                )
                epi_0_to_k.append(float(errs[0]))
        tr['epi_errs_0_to_k'] = epi_0_to_k  # start_id!=0 の場合は空リスト

    # ---- 0→k track counts (connectivity only) ----
    # 「フレーム0からフレームkまで途切れずにつながっているトラック数」
    # 条件: start_id == 0 かつ length >= k+1 （= 0..k の各フレームに点がある）
    track_counts_0_to_k = [0] * bag_size
    for tid, tr in tracks.items():
        if tr['start_id'] != 0:
            continue
        L = len(tr['points'])
        # このトラックは k = 0..L-1 まで有効
        max_k = min(L - 1, bag_size - 1)
        for k in range(0, max_k + 1):
            track_counts_0_to_k[k] += 1

    # print 用
    print("  [track counts 0→k] (number of tracks that exist from frame 0 to frame k)")
    for k, c in enumerate(track_counts_0_to_k):
        print(f"    k={k}: n_tracks_0_to_{k}={c}")

    return {
        'bag_file': str(bag_file),
        'image_paths': [str(p) for p in img_paths],
        'tracks': {str(tid): tr for tid, tr in tracks.items()},
        'track_counts_0_to_k': track_counts_0_to_k,
    }


# ============================================================
# 5) Main
# ============================================================

def main():
    args = parse_args()

    # JamMa config
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    profiler = build_profiler(args.profiler_name)

    config.JAMMA.USE_DET = True
    jamma_model = PL_JamMa(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        dump_dir=args.dump_dir,
    )

    config.JAMMA.USE_DET = False
    jamma_legacy_model = PL_JamMa(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        dump_dir=args.dump_dir,
    )
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # SP+LG
    sp_model = SuperPoint(max_num_keypoints=None, detection_threshold=0.0).eval().to(device)
    lg_model = LightGlue(
        features='superpoint',
        depth_confidence=-1,
        width_confidence=-1
    ).eval().to(device)

    # Bag list
    bag_files = sorted(args.subset_dir.glob(f"{args.bag_size}bag_*.txt"))
    if not bag_files:
        raise FileNotFoundError(
            f"No files found in {args.subset_dir} matching pattern {args.bag_size}bag_*.txt"
        )

    # 手法 → 関数
    methods: Dict[str, Callable[..., Any]] = {
        "jamma": run_jamma_pair,
        "splg": run_splg_pair,
        "jamma_legacy": run_jamma_pair,
    }
    methods_kwargs: Dict[str, dict] = {
        "jamma": {"jamma": jamma_model},
        "splg": {"sp_model": sp_model, "lg_model": lg_model},
        "jamma_legacy": {"jamma": jamma_legacy_model},
    }

    # 手法ごとに JSON を分ける
    for method_name, run_pair_fn in methods.items():
        print(f"\n=== Method: {method_name} ===")
        per_bag_results = []

        for bf in bag_files[0:1]:  # 1bagだけ処理する
            print(f"\n[Bag] {bf.name}")
            res = evaluate_bag_and_collect_tracks(
                bag_file=bf,
                bag_size=args.bag_size,
                dataset_root=args.dataset_root,
                calib_dir=args.calib_dir,
                flip_w2c=args.flip_w2c,
                run_pair_fn=run_pair_fn,
                run_pair_kwargs=methods_kwargs[method_name],
                device=device,
                method_name=method_name,
            )
            per_bag_results.append(res)

        # 保存パス
        out_path = args.save_json_base.parent / f"{args.save_json_base.stem}_{method_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({'per_bag': per_bag_results}, indent=2))
        print(f"\n[{method_name}] Tracks with epi errors saved to: {out_path}")


if __name__ == "__main__":
    main()
