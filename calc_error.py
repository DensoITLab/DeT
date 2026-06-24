#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate DeT-JamMa full-track validity against GT tracks.

What this script does:
1. Run sequential DeT-JamMa matching on bag images:
      0-1, 1-2, ..., N-2 -> N-1
2. Build tracks from pairwise matches
3. Keep only full tracks:
      start_id == 0, end_id == bag_size-1, len(points) == bag_size
4. For each full track:
      - take frame-0 point
      - lift it to 3D using depth0 + K0
      - project the same 3D point to every frame k using GT poses
      - compare projected GT point with DeT-JamMa tracked point
5. Report:
      - per-bag mean error per frame
      - global mean error per frame across all bags
        * avg_of_bag_means
        * pooled_mean_over_all_points

Outputs:
  --out_csv_per_bag      : one row per (bag, frame)
  --out_csv_global       : one row per frame
  --out_json             : detailed JSON

Notes:
- Assumes calibration R,t are world->camera (w2c), unless --flip_w2c is set.
- GT track is generated ONLY from frame-0 depth.
- Error is simple 2D Euclidean distance in pixels.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.dataset import read_megadepth_color
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler


# ============================================================
# Camera / depth IO
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

    return K.reshape(3, 3), R.reshape(3, 3), T.reshape(3)


def load_cam_from_dir(calib_dir: Path, img_path: Path, flip_w2c: bool) -> CameraParams:
    h5_path = calib_dir / f"calibration_{img_path.stem}.h5"
    K, R, t = _read_cam_from_h5(h5_path)
    if flip_w2c:
        R, t = R.T, -R.T @ t
    return CameraParams(K=K.astype(np.float64), R=R.astype(np.float64), t=t.astype(np.float64))


def _read_depth_from_h5(h5_path: Path) -> np.ndarray:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        if "depth" in f:
            depth = np.array(f["depth"])
        else:
            depth = None
            for key in f.keys():
                arr = np.array(f[key])
                if arr.ndim == 2:
                    depth = arr
                    break
            if depth is None:
                raise KeyError(f"depth dataset not found in {h5_path}")

    return depth.astype(np.float32)


def load_depth_from_dir(depth_dir: Path, img_path: Path) -> np.ndarray:
    return _read_depth_from_h5(depth_dir / f"{img_path.stem}.h5")


def read_bag_paths(bag_file: Path) -> List[str]:
    return [ln.strip() for ln in bag_file.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_image_size(img_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]
    return h, w


# ============================================================
# JamMa runner
# ============================================================

def build_jamma_model(
    main_cfg_path: str,
    data_cfg_path: str,
    profiler_name: str,
    ckpt_path: str,
    device: torch.device,
) -> PL_JamMa:
    config = get_cfg_defaults()
    config.merge_from_file(main_cfg_path)
    config.merge_from_file(data_cfg_path)

    config.JAMMA.DET.SEARCH_RADIUS = 832 * 2**0.5
    config.JAMMA.DET.FINE_THR = 0.0
    config.JAMMA.USE_COMPILE = False
    config.JAMMA.DET.USE_DET = True

    profiler = build_profiler(profiler_name)
    model = PL_JamMa(config, pretrained_ckpt=ckpt_path, profiler=profiler)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def run_jamma_pair(
    jamma: PL_JamMa,
    device: torch.device,
    imgA: Path,
    imgB: Path,
    prev_data: Any = None,
    image_idA: int = 0,
    image_idB: int = 1,
):
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

    result, _, _ = jamma(data)

    if "prev_data" in data:
        del data["prev_data"]

    mk0 = result["mkpts0_f_origin"]
    mk1 = result["mkpts1_f_origin"]
    mconf = result.get("mconf_f", None)

    return mk0, mk1, mconf, result


# ============================================================
# Track building (DeT-JamMa only / default linking only)
# ============================================================

def build_tracks_from_sequential_pairs(
    pair_matches: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    topk: int = 20000,
) -> Dict[int, dict]:
    tracks: Dict[int, dict] = {}
    next_tid = 0

    prev_point_to_tid: Dict[tuple, int] = {}

    for i, (mk0, mk1, mconf) in enumerate(pair_matches):
        mk0 = np.asarray(mk0, dtype=np.float32)
        mk1 = np.asarray(mk1, dtype=np.float32)

        if mconf is not None and len(mconf) > 0:
            mconf = np.asarray(mconf, dtype=np.float32)
            k = min(int(len(mconf)), topk)
            idx = np.argsort(-mconf)[:k]
            mk0 = mk0[idx]
            mk1 = mk1[idx]
            mconf = mconf[idx]
        else:
            mconf = None
            if len(mk0) > topk:
                mk0 = mk0[:topk]
                mk1 = mk1[:topk]

        best_per_A: Dict[tuple, dict] = {}
        for j, (ptA, ptB) in enumerate(zip(mk0, mk1)):
            keyA = (float(ptA[0]), float(ptA[1]))
            conf_val = float(mconf[j]) if mconf is not None else 1.0
            if keyA not in best_per_A or conf_val > best_per_A[keyA]["conf"]:
                best_per_A[keyA] = {"ptA": ptA, "ptB": ptB, "conf": conf_val}

        curr_point_to_tid: Dict[tuple, int] = {}
        for keyA, rec in best_per_A.items():
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

        prev_point_to_tid = curr_point_to_tid

    return tracks


def filter_full_length_tracks(tracks: Dict[int, dict], bag_size: int) -> Dict[int, dict]:
    out = {}
    for tid, tr in tracks.items():
        if (
            int(tr["start_id"]) == 0
            and int(tr["end_id"]) == bag_size - 1
            and len(tr["points"]) == bag_size
        ):
            out[tid] = tr
    return out


# ============================================================
# GT projection from frame-0 point
# ============================================================

def lift_point_to_world_from_depth(
    p0: np.ndarray,
    depth0: np.ndarray,
    cam0: CameraParams,
) -> Tuple[Optional[np.ndarray], bool]:
    x, y = float(p0[0]), float(p0[1])
    h, w = depth0.shape

    ix = int(round(x))
    iy = int(round(y))
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None, False

    z = float(depth0[iy, ix])
    if (not np.isfinite(z)) or z <= 0:
        return None, False

    K0_inv = np.linalg.inv(cam0.K)
    p_h = np.array([x, y, 1.0], dtype=np.float64)
    X_cam0 = K0_inv @ (p_h * z)

    X_world = cam0.R.T @ (X_cam0 - cam0.t.reshape(3,))
    return X_world.astype(np.float64), True


def project_world_to_image(
    X_world: np.ndarray,
    cam: CameraParams,
    image_hw: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], bool]:
    h, w = image_hw

    X_cam = cam.R @ X_world + cam.t.reshape(3,)
    z = float(X_cam[2])
    if z <= 0 or (not np.isfinite(z)):
        return None, False

    p = cam.K @ (X_cam / z)
    u = float(p[0])
    v = float(p[1])

    if u < 0 or u >= w or v < 0 or v >= h:
        return None, False

    return np.array([u, v], dtype=np.float64), True


# ============================================================
# Per-bag evaluation
# ============================================================

def evaluate_bag_gt_track_errors(
    bag_file: Path,
    bag_size: int,
    dataset_root: Path,
    calib_dir: Path,
    depth_dir: Path,
    flip_w2c: bool,
    jamma: PL_JamMa,
    device: torch.device,
    topk: int,
) -> Dict[str, Any]:
    rel_paths = read_bag_paths(bag_file)
    if len(rel_paths) != bag_size:
        raise ValueError(f"{bag_file}: expected {bag_size} images, got {len(rel_paths)}")

    img_paths = [dataset_root / rp for rp in rel_paths]
    cams = [load_cam_from_dir(calib_dir, p, flip_w2c) for p in img_paths]
    depths = [load_depth_from_dir(depth_dir, p) for p in img_paths]
    image_sizes = [read_image_size(p) for p in img_paths]

    prev_data = None
    pair_matches = []
    image_idA = 0
    image_idB = 1

    for i in range(bag_size - 1):
        mk0_t, mk1_t, mconf_t, prev_result = run_jamma_pair(
            jamma=jamma,
            device=device,
            imgA=img_paths[i],
            imgB=img_paths[i + 1],
            prev_data=prev_data,
            image_idA=image_idA,
            image_idB=image_idB,
        )
        prev_data = prev_result
        image_idA += 1
        image_idB += 1

        mk0 = mk0_t.detach().cpu().numpy() if hasattr(mk0_t, "detach") else np.asarray(mk0_t)
        mk1 = mk1_t.detach().cpu().numpy() if hasattr(mk1_t, "detach") else np.asarray(mk1_t)
        mconf = (
            mconf_t.detach().cpu().numpy()
            if (mconf_t is not None and hasattr(mconf_t, "detach"))
            else (None if mconf_t is None else np.asarray(mconf_t))
        )
        pair_matches.append((mk0, mk1, mconf))

    tracks_all = build_tracks_from_sequential_pairs(
        pair_matches=pair_matches,
        topk=topk,
    )
    tracks = filter_full_length_tracks(tracks_all, bag_size=bag_size)

    per_frame_errors: Dict[int, List[float]] = {k: [] for k in range(bag_size)}
    per_track_records = []

    for tid, tr in tracks.items():
        pts = np.asarray(tr["points"], dtype=np.float64)
        p0 = pts[0]

        X_world, ok0 = lift_point_to_world_from_depth(
            p0=p0,
            depth0=depths[0],
            cam0=cams[0],
        )
        if not ok0:
            continue

        track_errors = []
        track_valid = []

        for k in range(bag_size):
            gt_pt, valid = project_world_to_image(
                X_world=X_world,
                cam=cams[k],
                image_hw=image_sizes[k],
            )
            if not valid:
                track_errors.append(None)
                track_valid.append(False)
                continue

            det_pt = pts[k]
            err = float(np.linalg.norm(det_pt - gt_pt))

            per_frame_errors[k].append(err)
            track_errors.append(err)
            track_valid.append(True)

        per_track_records.append({
            "tid": int(tid),
            "errors_px": track_errors,
            "valid": track_valid,
        })

    bag_frame_stats = []
    for k in range(bag_size):
        errs = np.asarray(per_frame_errors[k], dtype=np.float64)
        if errs.size > 0:
            mean_err = float(np.mean(errs))
            median_err = float(np.median(errs))
            n_valid = int(errs.size)
        else:
            mean_err = float("nan")
            median_err = float("nan")
            n_valid = 0

        bag_frame_stats.append({
            "frame": int(k),
            "mean_err_px": mean_err,
            "median_err_px": median_err,
            "n_valid": n_valid,
        })

    return {
        "bag": bag_file.stem,
        "bag_file": str(bag_file),
        "num_tracks_all": int(len(tracks_all)),
        "num_full_tracks": int(len(tracks)),
        "num_tracks_used_for_gt_eval": int(len(per_track_records)),
        "per_frame_stats": bag_frame_stats,
        "per_track_records": per_track_records,
    }


# ============================================================
# CSV / JSON writers
# ============================================================

def write_csv_per_bag(path: Path, bag_results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("bag,frame,mean_err_px,median_err_px,n_valid,num_tracks_all,num_full_tracks,num_tracks_used_for_gt_eval\n")
        for br in bag_results:
            for st in br["per_frame_stats"]:
                f.write(
                    f"{br['bag']},"
                    f"{st['frame']},"
                    f"{st['mean_err_px']},"
                    f"{st['median_err_px']},"
                    f"{st['n_valid']},"
                    f"{br['num_tracks_all']},"
                    f"{br['num_full_tracks']},"
                    f"{br['num_tracks_used_for_gt_eval']}\n"
                )


def write_csv_global(path: Path, global_stats: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("frame,avg_of_bag_means_px,pooled_mean_px,pooled_median_px,num_bags_with_valid,num_points_total\n")
        for st in global_stats:
            f.write(
                f"{st['frame']},"
                f"{st['avg_of_bag_means_px']},"
                f"{st['pooled_mean_px']},"
                f"{st['pooled_median_px']},"
                f"{st['num_bags_with_valid']},"
                f"{st['num_points_total']}\n"
            )


# ============================================================
# Main
# ============================================================

def parse_args():
    default_root = Path("/home/ach17765lb/data/phototourism")
    scene = "st_peters_square"
    set_name = "set_100"

    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_dir", type=Path, default=default_root / scene / set_name / "sub_set")
    ap.add_argument("--dataset_root", type=Path, default=default_root / scene / set_name)
    ap.add_argument("--calib_dir", type=Path, default=default_root / scene / set_name / "calibration")
    ap.add_argument("--depth_dir", type=Path, default=default_root / scene / set_name / "depth_maps")

    ap.add_argument("--data_cfg_path", type=str, default="configs/data/megadepth_test_1500.py")
    ap.add_argument("--main_cfg_path", type=str, default="configs/jamma/outdoor/test.py")
    ap.add_argument("--profiler_name", type=str, default="inference")
    ap.add_argument("--ckpt_path", type=str, default="official")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bag_size", type=int, default=5)
    ap.add_argument("--flip_w2c", action="store_true")

    ap.add_argument("--topk", type=int, default=20000)

    ap.add_argument("--out_csv_per_bag", type=Path, default=Path("./det_jamma_gt_track_error_per_bag.csv"))
    ap.add_argument("--out_csv_global", type=Path, default=Path("./det_jamma_gt_track_error_global.csv"))
    ap.add_argument("--out_json", type=Path, default=Path("./det_jamma_gt_track_error_summary.json"))

    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    jamma = build_jamma_model(
        main_cfg_path=args.main_cfg_path,
        data_cfg_path=args.data_cfg_path,
        profiler_name=args.profiler_name,
        ckpt_path=args.ckpt_path,
        device=device,
    )
    print("[matcher] det-jamma")

    bag_files = sorted(args.subset_dir.glob(f"{args.bag_size}bag_*.txt"))
    if not bag_files:
        raise FileNotFoundError(f"No bag files found in {args.subset_dir}")

    bag_results = []
    pooled_errors_per_frame: Dict[int, List[float]] = {k: [] for k in range(args.bag_size)}
    bag_means_per_frame: Dict[int, List[float]] = {k: [] for k in range(args.bag_size)}

    for bf in bag_files:
        print(f"\n[Bag] {bf.name}")
        res = evaluate_bag_gt_track_errors(
            bag_file=bf,
            bag_size=args.bag_size,
            dataset_root=args.dataset_root,
            calib_dir=args.calib_dir,
            depth_dir=args.depth_dir,
            flip_w2c=args.flip_w2c,
            jamma=jamma,
            device=device,
            topk=args.topk,
        )

        print(
            f"  all_tracks={res['num_tracks_all']} "
            f"full_tracks={res['num_full_tracks']} "
            f"used_for_gt_eval={res['num_tracks_used_for_gt_eval']}"
        )

        for st in res["per_frame_stats"]:
            k = st["frame"]
            print(
                f"  frame={k}: mean_err_px={st['mean_err_px']:.4f} "
                f"median_err_px={st['median_err_px']:.4f} "
                f"n_valid={st['n_valid']}"
            )

            if np.isfinite(st["mean_err_px"]):
                bag_means_per_frame[k].append(float(st["mean_err_px"]))

        for tr in res["per_track_records"]:
            errs = tr["errors_px"]
            valids = tr["valid"]
            for k in range(args.bag_size):
                if valids[k] and errs[k] is not None:
                    pooled_errors_per_frame[k].append(float(errs[k]))

        bag_results.append(res)

    global_stats = []
    print("\n=== Global frame-wise GT-track error summary ===")
    for k in range(args.bag_size):
        pooled = np.asarray(pooled_errors_per_frame[k], dtype=np.float64)
        bagmeans = np.asarray(bag_means_per_frame[k], dtype=np.float64)

        pooled_mean = float(np.mean(pooled)) if pooled.size > 0 else float("nan")
        pooled_median = float(np.median(pooled)) if pooled.size > 0 else float("nan")
        avg_of_bag_means = float(np.mean(bagmeans)) if bagmeans.size > 0 else float("nan")

        st = {
            "frame": int(k),
            "avg_of_bag_means_px": avg_of_bag_means,
            "pooled_mean_px": pooled_mean,
            "pooled_median_px": pooled_median,
            "num_bags_with_valid": int(bagmeans.size),
            "num_points_total": int(pooled.size),
        }
        global_stats.append(st)

        print(
            f"frame={k}: "
            f"avg_of_bag_means_px={avg_of_bag_means:.4f}, "
            f"pooled_mean_px={pooled_mean:.4f}, "
            f"pooled_median_px={pooled_median:.4f}, "
            f"num_bags_with_valid={bagmeans.size}, "
            f"num_points_total={pooled.size}"
        )

    write_csv_per_bag(args.out_csv_per_bag, bag_results)
    write_csv_global(args.out_csv_global, global_stats)

    payload = {
        "method": "det-jamma",
        "bag_size": int(args.bag_size),
        "num_bags": int(len(bag_results)),
        "global_stats": global_stats,
        "bags": bag_results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved per-bag CSV : {args.out_csv_per_bag}")
    print(f"Saved global CSV  : {args.out_csv_global}")
    print(f"Saved JSON        : {args.out_json}")


if __name__ == "__main__":
    main()