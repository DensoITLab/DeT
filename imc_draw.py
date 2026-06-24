#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ONE bag file, match adjacent pairs, link 0->N tracks, then for each edge
(i,i+1) compute:
  ① symmetric epipolar error per track segment
  ② rotation/translation error (edge-level, from LO-RANSAC pose estimation)
  ③ depth reprojection error [px] per track segment (one-side)

Then draw (N+1) images concatenated horizontally and color each track segment:
  green if (epi_ok AND pose_ok AND reproj_ok) else red.

Outputs a single visualization PNG.

Dependencies:
- JamMa project-local modules (src.*) if --method jamma / jamma_legacy
- lightglue if --method splg
- opencv-python, numpy, torch, h5py
"""

import argparse
import dataclasses
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- JamMa imports (project-local) ---
from src.utils.dataset import read_megadepth_color
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler

# LO-RANSAC / pose / epi utilities (JamMa実装)
from src.utils.metrics import (
    estimate_lo_pose,
    relative_pose_error,
    symmetric_epipolar_distance,
)

# --- LightGlue imports ---
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

from thop import profile

# --- Dataset defaults (PhotoTourism example) ---
default_root = Path('/home/ach17765lb/data/phototourism')
scene = 'st_peters_square'   # 'reichstag' , 'sacre_coeur', 'st_peters_square'
set_name = 'set_100'
subset_size = 5


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- single bag ---
    p.add_argument("--bag_file", type=Path, default= default_root / scene / set_name / 'sub_set' / f"{subset_size}bag_000.txt",
                   help="Path to bag file listing image paths")
    p.add_argument("--bag_size", type=int, default=subset_size, help="Number of images in the bag file")

    # --- dataset ---
    
    p.add_argument('--dataset_root', type=Path, default=default_root / scene / set_name,
                        help='Root to prepend to image relative paths in bag files')
    p.add_argument('--calib_dir', type=Path, default=default_root / scene / set_name / 'calibration',
                        help='Directory with calibration_<stem>.h5 per image')
    p.add_argument('--depth_dir', type=Path, default=default_root / scene / set_name / 'depth_maps',
                        help='Directory with <stem>.h5 per image (depth)')


    p.add_argument("--flip_w2c", action="store_true",
                   help="If calib is world->cam, convert to cam->world internally")

    # --- method ---
    p.add_argument("--method", type=str, default="jamma",
                   choices=["jamma", "jamma_legacy", "splg"],
                   help="Matcher method to run")

    # --- JamMa config ---
    p.add_argument("--data_cfg_path", type=str, default="configs/data/megadepth_test_1500.py")
    p.add_argument("--main_cfg_path", type=str, default="configs/jamma/outdoor/test.py")
    p.add_argument("--ckpt_path", type=str, default="official")
    p.add_argument("--dump_dir", type=str, default="dump/jamma_outdoor")
    p.add_argument("--profiler_name", type=str, default="inference")

    # --- runtime ---
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--topk", type=int, default=20000)

    # --- thresholds ---
    p.add_argument("--epi_thr", type=float, default=1e-4, help="Epipolar error threshold")
    p.add_argument("--pose_thr_deg", type=float, default=5.0,
                   help="Pose threshold: max(R_err, t_err) [deg]")
    p.add_argument("--reproj_thr_px", type=float, default=3.0,
                   help="Depth reprojection threshold [px] (one-side)")

    # --- visualization ---
    p.add_argument("--line_thickness", type=int, default=2)
    p.add_argument("--circle_radius", type=int, default=3)
    p.add_argument("--max_tracks_draw", type=int, default=5000,
                   help="Limit number of tracks drawn for readability/perf")
    p.add_argument("--out_vis", type=Path, default=Path("tracks_vis_jamma.png"))
        # --- which metrics to use for visualization thresholding ---
    p.add_argument(
        "--vis_metrics",
        type=str,
        default="epi",
        help="Comma-separated metrics used for coloring. choose from: epi,pose,reproj. "
             "Example: --vis_metrics epi,reproj"
    )


    return p.parse_args()


# ============================================================
# Calibration / Depth utils
# ============================================================

@dataclasses.dataclass
class CameraParams:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


def _read_cam_from_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "K" in f and "R" in f and ("T" in f or "t" in f):
            K = np.array(f["K"])
            R = np.array(f["R"])
            T = np.array(f["T"] if "T" in f else f["t"])
        else:
            K = R = T = None
            for key in f.keys():
                g = f[key]
                if isinstance(g, h5py.Group) and {"K", "R"}.issubset(g.keys()) and ("T" in g or "t" in g):
                    K = np.array(g["K"])
                    R = np.array(g["R"])
                    T = np.array(g["T"] if "T" in g else g["t"])
                    break
            if K is None or R is None or T is None:
                raise KeyError(f"K/R/T not found in {h5_path}")
    return K.reshape(3, 3), R.reshape(3, 3), T.reshape(3,)


def load_cam_from_dir(calib_dir: Path, img_path: Path, flip_w2c: bool) -> CameraParams:
    h5_path = calib_dir / f"calibration_{img_path.stem}.h5"
    K, R, t = _read_cam_from_h5(h5_path)
    if flip_w2c:
        R, t = R.T, -R.T @ t
    return CameraParams(K=K, R=R, t=t)


def _read_depth_from_h5(h5_path: Path) -> np.ndarray:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "depth" in f:
            depth = np.array(f["depth"])
        else:
            depth = None
            for key in f.keys():
                d = np.array(f[key])
                if d.ndim == 2:
                    depth = d
                    break
            if depth is None:
                raise KeyError(f"depth dataset not found in {h5_path}")
    return depth.astype(np.float32)


def load_depth_from_dir(depth_dir: Path, img_path: Path) -> np.ndarray:
    h5_path = depth_dir / f"{img_path.stem}.h5"
    return _read_depth_from_h5(h5_path)


def _relative_pose(camA: CameraParams, camB: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
    """Relative pose of camB wrt camA."""
    R21 = camB.R @ camA.R.T
    t21 = camB.t - R21 @ camA.t
    return R21, t21


def _compute_symmetric_epi_error(
    x0_px: np.ndarray, x1_px: np.ndarray, cam0: CameraParams, cam1: CameraParams, device: torch.device
) -> np.ndarray:
    """symmetric epipolar distance per correspondence."""
    if x0_px.size == 0:
        return np.zeros((0,), dtype=np.float64)

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


def _depth_reproj_one_side_px(
    p_src: np.ndarray,
    p_tgt: np.ndarray,
    depth_src: Optional[np.ndarray],
    cam_src: CameraParams,
    cam_tgt: CameraParams,
) -> Tuple[float, bool]:
    """
    One-side depth reprojection error:
    - unproject p_src using depth_src into 3D
    - project into target view -> predicted p_tgt_hat
    - return ||p_tgt - p_tgt_hat|| in px
    """
    if depth_src is None:
        return 0.0, False

    Hs, Ws = depth_src.shape
    xs, ys = float(p_src[0]), float(p_src[1])
    ix = int(round(xs))
    iy = int(round(ys))
    if ix < 0 or ix >= Ws or iy < 0 or iy >= Hs:
        return 0.0, False

    Z = float(depth_src[iy, ix])
    if not np.isfinite(Z) or Z <= 0:
        return 0.0, False

    Kinv = np.linalg.inv(cam_src.K)
    R_s, t_s = cam_src.R, cam_src.t.reshape(3,)
    R_t, t_t = cam_tgt.R, cam_tgt.t.reshape(3,)

    p_h = np.array([xs, ys, 1.0], dtype=np.float32)
    X_cam_s = Kinv @ (p_h * Z)
    X_world = R_s.T @ (X_cam_s - t_s)

    X_cam_t = R_t @ X_world + t_t
    if X_cam_t[2] <= 0:
        return 0.0, False

    p_proj = cam_tgt.K @ (X_cam_t / X_cam_t[2])
    u, v = float(p_proj[0]), float(p_proj[1])

    err = float(np.linalg.norm(p_tgt - np.array([u, v], dtype=np.float32)))
    return err, True


# ============================================================
# Matchers
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


def run_jamma_pair(device: torch.device,
                   imgA: Path,
                   imgB: Path,
                   prev_data: Any = None,
                   image_idA: int = 0,
                   image_idB: int = 1,
                   jamma: PL_JamMa = None):
    if jamma is None:
        raise ValueError("jamma model must be provided")

    image0, s0, m0, p0, *_ = read_megadepth_color(str(imgA), 832, 8, True)
    image1, s1, m1, p1, *_ = read_megadepth_color(str(imgB), 832, 8, True)

    m0 = F.interpolate(m0[None, None].float(), scale_factor=0.125, mode="nearest")[0].bool()
    m1 = F.interpolate(m1[None, None].float(), scale_factor=0.125, mode="nearest")[0].bool()

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
        image_idA=image_idA,
        image_idB=image_idB,
    )
    if prev_data is not None:
        data["prev_data"] = prev_data

    jamma = jamma.to(device).eval()
    with torch.no_grad():
        result, flops, runtime = jamma(data)

    mk0 = result["mkpts0_f_origin"]
    mk1 = result["mkpts1_f_origin"]
    mconf = result.get("mconf_f", None)
    return mk0, mk1, mconf, float(flops), float(runtime), result


def run_splg_pair(device: torch.device,
                  imgA: Path,
                  imgB: Path,
                  prev_data: Any = None,
                  image_idA: int = 0,
                  image_idB: int = 1,
                  sp_model: Any = None,
                  lg_model: Any = None):
    if sp_model is None or lg_model is None:
        raise ValueError("sp_model and lg_model required")

    image0 = load_image(str(imgA)).to(device)
    image1 = load_image(str(imgB)).to(device)

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

    matches = matches01["matches"]
    points0 = feats0["keypoints"][matches[:, 0]]
    points1 = feats1["keypoints"][matches[:, 1]]
    mconf = matches01.get("scores", torch.ones(points0.shape[0], device=device, dtype=torch.float32))

    mk0 = points0.to(device).float()
    mk1 = points1.to(device).float()
    mconf = mconf.to(device).float()

    # FLOPs (optional-ish)
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

    return mk0, mk1, mconf, float(flops), float(runtime_ms), {}


# ============================================================
# Bag read + track linking
# ============================================================

def read_bag_paths(bag_file: Path) -> List[str]:
    return [ln.strip() for ln in bag_file.read_text(encoding="utf-8").splitlines() if ln.strip()]


def link_tracks_exact_prev_key(
    mk0: np.ndarray,
    mk1: np.ndarray,
    mconf: Optional[np.ndarray],
    i: int,
    tracks: Dict[int, dict],
    next_tid: int,
    prev_point_to_tid: Dict[Tuple[float, float], int],
) -> Tuple[int, Dict[Tuple[float, float], int]]:
    """
    Exact key-based linking (same as your non-legacy branch):
    - keep best match per keyA
    - key by float(x), float(y) (origin coords)
    """
    curr_point_to_tid: Dict[Tuple[float, float], int] = {}
    best_matches_per_keyA: Dict[Tuple[float, float], dict] = {}

    for idx_match, (ptA, ptB) in enumerate(zip(mk0, mk1)):
        keyA = (float(ptA[0]), float(ptA[1]))
        conf_val = float(mconf[idx_match]) if mconf is not None else 1.0
        if keyA not in best_matches_per_keyA or conf_val > best_matches_per_keyA[keyA]["conf"]:
            best_matches_per_keyA[keyA] = {"ptA": ptA, "ptB": ptB, "conf": conf_val}

    for keyA, rec in best_matches_per_keyA.items():
        ptA = rec["ptA"]
        ptB = rec["ptB"]
        conf_val = rec["conf"]
        keyB = (float(ptB[0]), float(ptB[1]))

        if keyA in prev_point_to_tid:
            tid = prev_point_to_tid[keyA]
            tr = tracks[tid]
            tr["points"].append([float(ptB[0]), float(ptB[1])])
            tr["end_id"] = i + 1
            tr["confs"].append(conf_val)
        else:
            tid = next_tid
            next_tid += 1
            tracks[tid] = {
                "start_id": i,
                "end_id": i + 1,
                "points": [
                    [float(ptA[0]), float(ptA[1])],
                    [float(ptB[0]), float(ptB[1])],
                ],
                "confs": [conf_val],
            }

        curr_point_to_tid[keyB] = tid

    return next_tid, curr_point_to_tid


# ============================================================
# Visualization helpers
# ============================================================

def load_bgr_image(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return im


def concat_images_horiz(images: List[np.ndarray]) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Return concatenated canvas, x_offsets, widths.
    Images are placed at top-left with no scaling; canvas height = max height.
    """
    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    H = int(max(heights))
    W = int(sum(widths))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    x_offsets = []
    x = 0
    for im in images:
        h, w = im.shape[:2]
        canvas[0:h, x:x+w] = im
        x_offsets.append(x)
        x += w

    return canvas, x_offsets, widths

def normalize_vis_metrics(vis_metrics_str: str) -> List[str]:
    allowed = {"epi", "pose", "reproj"}
    ms = [m.strip().lower() for m in (vis_metrics_str or "").split(",") if m.strip()]
    ms = [m for m in ms if m in allowed]
    if len(ms) == 0:
        ms = ["epi", "pose", "reproj"]
    return ms

def compute_edge_green_counts(
    tracks: Dict[int, dict],
    bag_size: int,
    epi_ok: Dict[Tuple[int, int, int], bool],
    reproj_ok: Dict[Tuple[int, int, int], bool],
    pose_ok_edge: Dict[Tuple[int, int], bool],
    vis_metrics: List[str],
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Return {(fi,fj): (green_count, total_count)} for each adjacent edge.
    """
    counts: Dict[Tuple[int, int], List[int]] = {}  # edge -> [green, total]
    for i in range(bag_size - 1):
        counts[(i, i + 1)] = [0, 0]

    for tid, tr in tracks.items():
        s = int(tr["start_id"])
        pts = tr.get("points", [])
        for local_i in range(len(pts) - 1):
            fi = s + local_i
            fj = fi + 1
            if fj >= bag_size:
                continue

            key_seg = (tid, fi, fj)
            key_edge = (fi, fj)

            ok_map = {
                "epi": bool(epi_ok.get(key_seg, False)),
                "reproj": bool(reproj_ok.get(key_seg, False)),
                "pose": bool(pose_ok_edge.get(key_edge, False)),
            }
            ok_all = all(ok_map[m] for m in vis_metrics)

            counts[key_edge][1] += 1
            if ok_all:
                counts[key_edge][0] += 1

    return {k: (v[0], v[1]) for k, v in counts.items()}


def draw_colored_tracks(
    canvas: np.ndarray,
    tracks: Dict[int, dict],
    x_offsets: List[int],
    epi_ok: Dict[Tuple[int, int, int], bool],
    reproj_ok: Dict[Tuple[int, int, int], bool],
    pose_ok_edge: Dict[Tuple[int, int], bool],
    vis_metrics: List[str],
    line_thickness: int = 2,
    circle_radius: int = 3,
    max_tracks_draw: int = 5000,
) -> np.ndarray:
    green = (0, 255, 0)
    red = (0, 0, 255)

    vis_metrics = [m.strip().lower() for m in vis_metrics if m.strip()]
    allowed = {"epi", "pose", "reproj"}
    vis_metrics = [m for m in vis_metrics if m in allowed]
    if len(vis_metrics) == 0:
        # 何も指定されてない/不正なら全採用にフォールバック
        vis_metrics = ["epi", "pose", "reproj"]

    tids = sorted(tracks.keys())
    if len(tids) > max_tracks_draw:
        tids = tids[:max_tracks_draw]

    for tid in tids:
        tr = tracks[tid]
        pts = tr.get("points", [])
        start_id = int(tr.get("start_id", 0))

        for local_i in range(len(pts) - 1):
            fi = start_id + local_i
            fj = fi + 1
            if fj >= len(x_offsets):
                continue

            p_i = np.array(pts[local_i], dtype=np.float32)
            p_j = np.array(pts[local_i + 1], dtype=np.float32)

            key_seg = (tid, fi, fj)
            key_edge = (fi, fj)

            ok_map = {
                "epi": bool(epi_ok.get(key_seg, False)),
                "reproj": bool(reproj_ok.get(key_seg, False)),
                "pose": bool(pose_ok_edge.get(key_edge, False)),
            }
            ok_all = all(ok_map[m] for m in vis_metrics)

            color = green if ok_all else red

            xi = int(round(p_i[0] + x_offsets[fi]))
            yi = int(round(p_i[1]))
            xj = int(round(p_j[0] + x_offsets[fj]))
            yj = int(round(p_j[1]))

            cv2.line(canvas, (xi, yi), (xj, yj), color, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (xi, yi), circle_radius, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (xj, yj), circle_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas



def filter_full_tracks_0_to_N(tracks: Dict[int, dict], bag_size: int) -> Dict[int, dict]:
    """
    Keep only tracks that start at frame 0 and reach frame N (=bag_size-1),
    with exactly bag_size points (one per frame).
    """
    num_pairs = bag_size - 1
    full = {}
    for tid, tr in tracks.items():
        if int(tr.get("start_id", -1)) != 0:
            continue
        if int(tr.get("end_id", -1)) != num_pairs:
            continue
        pts = tr.get("points", [])
        if len(pts) != bag_size:
            continue
        full[tid] = tr
    return full


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    rel_paths = read_bag_paths(args.bag_file)
    if len(rel_paths) != args.bag_size:
        raise ValueError(f"{args.bag_file}: expected {args.bag_size} paths, got {len(rel_paths)}")

    img_paths = [args.dataset_root / rp for rp in rel_paths]
    for p in img_paths:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

    cams = [load_cam_from_dir(args.calib_dir, p, args.flip_w2c) for p in img_paths]

    depths: List[Optional[np.ndarray]] = [None] * args.bag_size
    if args.depth_dir is not None:
        depth_dir = Path(args.depth_dir)
        for i, p in enumerate(img_paths):
            try:
                depths[i] = load_depth_from_dir(depth_dir, p)
            except Exception as e:
                print(f"[WARN] depth load failed for {p}: {e}")
                depths[i] = None

    # --- init method ---
    jamma_model = None
    jamma_legacy_model = None
    sp_model = None
    lg_model = None

    if args.method in ["jamma", "jamma_legacy"]:
        config = get_cfg_defaults()
        config.merge_from_file(args.main_cfg_path)
        config.merge_from_file(args.data_cfg_path)

        # mimic your setup
        config.JAMMA.DET.SEARCH_RADIUS = 832 * 2**0.5
        config.JAMMA.DET.FINE_THR = 0.0
        config.JAMMA.USE_COMPILE = False

        profiler = build_profiler(args.profiler_name)

        if args.method == "jamma":
            jamma_model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path,
                                  profiler=profiler, dump_dir=args.dump_dir)
        else:
            # legacy model toggle
            config.JAMMA.DET.USE_DET = False
            jamma_legacy_model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path,
                                          profiler=profiler, dump_dir=args.dump_dir)

    elif args.method == "splg":
        sp_model = SuperPoint(max_num_keypoints=None, detection_threshold=0.0).eval().to(device)
        lg_model = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().to(device)

    # --- run matching on adjacent pairs + link tracks ---
    tracks: Dict[int, dict] = {}
    next_tid = 0
    prev_point_to_tid: Dict[Tuple[float, float], int] = {}
    prev_data: Any = None

    num_pairs = args.bag_size - 1
    image_idA = 0
    image_idB = 1

    for i in range(num_pairs):
        imgA = img_paths[i]
        imgB = img_paths[i + 1]
        print(f"[Pair {i}] {imgA.name} <-> {imgB.name}")

        if args.method == "jamma":
            mk0_t, mk1_t, mconf_t, _, _, prev_result = run_jamma_pair(
                device=device, imgA=imgA, imgB=imgB, prev_data=prev_data,
                image_idA=image_idA, image_idB=image_idB, jamma=jamma_model
            )
            prev_data = prev_result
        elif args.method == "jamma_legacy":
            mk0_t, mk1_t, mconf_t, _, _, prev_result = run_jamma_pair(
                device=device, imgA=imgA, imgB=imgB, prev_data=prev_data,
                image_idA=image_idA, image_idB=image_idB, jamma=jamma_legacy_model
            )
            prev_data = prev_result
        else:
            mk0_t, mk1_t, mconf_t, _, _, prev_result = run_splg_pair(
                device=device, imgA=imgA, imgB=imgB, prev_data=prev_data,
                image_idA=image_idA, image_idB=image_idB, sp_model=sp_model, lg_model=lg_model
            )
            prev_data = prev_result

        image_idA += 1
        image_idB += 1

        # topk by confidence
        mk0 = mk0_t.detach().cpu().numpy()
        mk1 = mk1_t.detach().cpu().numpy()
        if mconf_t is not None and mconf_t.numel() > 0:
            mconf = mconf_t.detach().cpu().numpy().astype(np.float32)
            k = min(int(mconf.size), int(args.topk))
            idx = np.argpartition(-mconf, kth=k-1)[:k]
            mk0 = mk0[idx]
            mk1 = mk1[idx]
            mconf = mconf[idx]
        else:
            mconf = None

        next_tid, prev_point_to_tid = link_tracks_exact_prev_key(
            mk0=mk0, mk1=mk1, mconf=mconf, i=i,
            tracks=tracks, next_tid=next_tid,
            prev_point_to_tid=prev_point_to_tid
        )
    tracks = filter_full_tracks_0_to_N(tracks, args.bag_size)
    print(f"[Tracks] full 0->N tracks: {len(tracks)}")


    # --- compute per-edge pose error (edge-level) ---
    pose_ok_edge: Dict[Tuple[int, int], bool] = {}
    pose_err_edge: Dict[Tuple[int, int], float] = {}

    pixel_thr = 0.5
    conf_ransac = 0.99999

    for i in range(num_pairs):
        fi, fj = i, i + 1

        # gather correspondences from tracks that have both frames
        pts_i = []
        pts_j = []
        for tr in tracks.values():
            s = int(tr["start_id"])
            e = int(tr["end_id"])
            pts = tr["points"]
            if not (s <= fi and e >= fj):
                continue
            li = fi - s
            lj = fj - s
            if li < 0 or lj < 0 or lj >= len(pts) or li >= len(pts):
                continue
            pts_i.append(pts[li])
            pts_j.append(pts[lj])

        pts_i = np.asarray(pts_i, dtype=np.float32)
        pts_j = np.asarray(pts_j, dtype=np.float32)

        if pts_i.shape[0] < 8:
            pose_ok_edge[(fi, fj)] = False
            pose_err_edge[(fi, fj)] = float("nan")
            print(f"[Edge {fi}->{fj}] pose: insufficient correspondences ({pts_i.shape[0]}) -> FAIL")
            continue

        est = estimate_lo_pose(pts_i, pts_j, cams[fi].K, cams[fj].K, pixel_thr, conf=conf_ransac)
        if not est.get("success", False):
            pose_ok_edge[(fi, fj)] = False
            pose_err_edge[(fi, fj)] = float("nan")
            print(f"[Edge {fi}->{fj}] pose: LO-RANSAC failed -> FAIL")
            continue

        M = est.get("M_0to1", None)
        if M is None:
            pose_ok_edge[(fi, fj)] = False
            pose_err_edge[(fi, fj)] = float("nan")
            print(f"[Edge {fi}->{fj}] pose: missing M_0to1 -> FAIL")
            continue

        R_gt, t_gt = _relative_pose(cams[fi], cams[fj])
        T_gt = np.block([[R_gt, t_gt.reshape(3, 1)],
                         [np.zeros((1, 3)), np.array([[1.0]])]])

        t_error, r_error = relative_pose_error(T_gt, M.R, M.t, ignore_gt_t_thr=0.0)
        pose_err = float(max(r_error, t_error))
        ok_pose = pose_err <= float(args.pose_thr_deg)

        pose_ok_edge[(fi, fj)] = bool(ok_pose)
        pose_err_edge[(fi, fj)] = pose_err
        print(f"[Edge {fi}->{fj}] pose_err(max(R,T))={pose_err:.2f} deg -> {'OK' if ok_pose else 'FAIL'}")

    # --- compute per-track segment epi/reproj ok ---
    epi_ok: Dict[Tuple[int, int, int], bool] = {}
    reproj_ok: Dict[Tuple[int, int, int], bool] = {}

    for tid, tr in tracks.items():
        s = int(tr["start_id"])
        pts = np.asarray(tr["points"], dtype=np.float32)
        for local_i in range(pts.shape[0] - 1):
            fi = s + local_i
            fj = fi + 1
            if fj >= args.bag_size:
                continue

            p_i = pts[local_i]
            p_j = pts[local_i + 1]

            # epi error per segment
            e = _compute_symmetric_epi_error(
                x0_px=p_i[None, :],
                x1_px=p_j[None, :],
                cam0=cams[fi],
                cam1=cams[fj],
                device=device
            )[0]
            ok_epi = bool(e < float(args.epi_thr))
            epi_ok[(tid, fi, fj)] = ok_epi

            # depth reprojection one-side (src=fi)
            err_px, valid = _depth_reproj_one_side_px(
                p_src=p_i, p_tgt=p_j,
                depth_src=depths[fi],
                cam_src=cams[fi], cam_tgt=cams[fj]
            )
            ok_rep = bool(valid and (err_px <= float(args.reproj_thr_px)))
            reproj_ok[(tid, fi, fj)] = ok_rep

    # --- load images and draw ---
    images_bgr = [load_bgr_image(p) for p in img_paths]
    canvas, x_offsets, _ = concat_images_horiz(images_bgr)
    #vis_metrics = [s.strip() for s in args.vis_metrics.split(",")]
    vis_metrics = normalize_vis_metrics(args.vis_metrics)

    edge_green = compute_edge_green_counts(
    tracks=tracks,
    bag_size=args.bag_size,
    epi_ok=epi_ok,
    reproj_ok=reproj_ok,
    pose_ok_edge=pose_ok_edge,
    vis_metrics=vis_metrics,
    )

    for i in range(num_pairs):
        fi, fj = i, i + 1
        g, t = edge_green.get((fi, fj), (0, 0))
        print(f"[Edge {fi}->{fj}] green {g}/{t} (use={','.join(vis_metrics)})")


    canvas = draw_colored_tracks(
        canvas=canvas,
        tracks=tracks,
        x_offsets=x_offsets,
        epi_ok=epi_ok,
        reproj_ok=reproj_ok,
        pose_ok_edge=pose_ok_edge,
        vis_metrics=vis_metrics,
        line_thickness=args.line_thickness,
        circle_radius=args.circle_radius,
        max_tracks_draw=args.max_tracks_draw,
    )
    # --- annotate thresholds summary on top-left ---
    txt = (f"method={args.method} | use={args.vis_metrics} | "
           f"epi<{args.epi_thr:g} | pose<{args.pose_thr_deg:g}deg | "
           f"reproj<{args.reproj_thr_px:g}px | tracks={len(tracks)}")
    cv2.putText(canvas, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    # edge pose status
    y = 60
    for i in range(num_pairs):
        fi, fj = i, i + 1
        pe = pose_err_edge.get((fi, fj), float("nan"))
        ok = pose_ok_edge.get((fi, fj), False)
        g, t = edge_green.get((fi, fj), (0, 0))
        s_line = (f"edge {fi}->{fj}: green={g}/{t} | "
          f"pose_err={pe:.2f} deg -> {'OK' if ok else 'FAIL'}")

        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(canvas, s_line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 25

    args.out_vis.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.out_vis), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save: {args.out_vis}")
    print(f"[Saved] {args.out_vis}")


if __name__ == "__main__":
    main()
