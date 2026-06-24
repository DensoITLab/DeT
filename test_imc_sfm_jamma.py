#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online SfM evaluation on IMC-like 5-image bags using sequential pair matches (0-1,1-2,2-3,3-4).

Metrics (per bag):
- Bootstrap success (E + recoverPose + triangulate from 0-1)
- PnP stability (success, inliers, median reproj) for k=2..N-1
- Map growth (#3D points after each step, #new triangulated)
- Final pose closeness to GT (0->N rot / t-dir error)
- RPE (edge k-1->k rot / t-dir error)
- Drift rate (deg/m) using GT path length (scale-safe)
- Tracking continuity (longest consecutive success streak, success rates)
- Runtime (matcher runtime sum/avg, SfM runtime breakdown)
- FLOPs (matcher FLOPs sum/avg; SfM FLOPs not computed)

Added metrics (global over all bags):
- AUC@K° (IMC/LoFTR-style): AUC of recall(error) from 0..K normalized by K
  where error = max(rotation_error_deg, translation_direction_error_deg)
  * "Multiview-like": final 0->N error per bag (FAILURES INCLUDED)
  * "Stereo-like": RPE edge error across all edges (FAILURES INCLUDED)
  * "Online-like": prefix 0->k error across all k in all bags (FAILURES INCLUDED)
- mAcc@K° (fraction under K; FAILURES INCLUDED)

IMPORTANT CHANGE (vs previous version):
- AUC/mAcc now INCLUDE failures (bootstrap failure, invalid poses/edges) as large errors.
  This avoids "only-valid" bias and better matches online tracking evaluation.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import cv2
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.dataset import read_megadepth_color
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler

# --- LightGlue ---
from lightglue import LightGlue, SuperPoint, SIFT
from lightglue.utils import load_image, rbd

# FLOPs
from thop import profile


# ============================================================
# Camera IO
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
        # if the file stores c2w, flip to w2c
        R, t = R.T, -R.T @ t
    return CameraParams(K, R, t)


def read_bag_paths(bag_file: Path) -> List[str]:
    return [ln.strip() for ln in bag_file.read_text(encoding='utf-8').splitlines() if ln.strip()]


# ============================================================
# Pose error utilities
# ============================================================

def _relative_pose_gt(camA: CameraParams, camB: CameraParams):
    """Relative pose of camB wrt camA. Assumes R,t are world->cam (w2c)."""
    R21 = camB.R @ camA.R.T
    t21 = camB.t - R21 @ camA.t
    return R21, t21


def _relative_pose_from_w2c(R_w2c_A: np.ndarray, t_w2c_A: np.ndarray,
                           R_w2c_B: np.ndarray, t_w2c_B: np.ndarray):
    """Relative pose of camB wrt camA, assuming both are w2c."""
    R_BA = R_w2c_B @ R_w2c_A.T
    t_BA = t_w2c_B - R_BA @ t_w2c_A
    return R_BA, t_BA


def _rot_err_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    dR = R_est @ R_gt.T
    cos = (np.trace(dR) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _t_dir_err_deg(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    tg = t_gt.reshape(3).astype(np.float64)
    te = t_est.reshape(3).astype(np.float64)
    ng = np.linalg.norm(tg) + 1e-12
    ne = np.linalg.norm(te) + 1e-12
    tg /= ng
    te /= ne
    cos = float(np.clip(np.dot(tg, te), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _norm(t: np.ndarray) -> float:
    return float(np.linalg.norm(t.reshape(-1)))


# ============================================================
# AUC metrics (IMC/LoFTR-style)  --- FAILURES INCLUDED ---
# ============================================================

def _pose_max_error_deg(R_err_deg: float, t_dir_err_deg: float) -> float:
    """IMC系の error 定義でよく使う: max(rot_err, trans_dir_err)"""
    return float(max(float(R_err_deg), float(t_dir_err_deg)))


def _clamp_failures(errors_deg: List[float], thresholds: Tuple[float, ...]) -> List[float]:
    """
    Replace NaN/Inf with a large value (treated as failure).
    This lets AUC/mAcc include failures instead of dropping them.
    """
    thrs = [float(t) for t in thresholds]
    big = max(thrs) * 10.0 + 1.0
    out = []
    for e in errors_deg:
        try:
            v = float(e)
        except Exception:
            v = float("nan")
        if not np.isfinite(v):
            v = big
        out.append(v)
    return out


def auc_pose_errors(errors_deg: List[float], thresholds=(5, 10, 20)) -> Dict[str, float]:
    """
    AUC@K = (∫_0^K recall(e) de) / K
    - errors_deg: list of scalar errors (deg). Non-finite are treated as failures.
    - returns 0..1 (multiply by 100 for %)
    """
    thresholds = tuple(float(t) for t in thresholds)
    if len(errors_deg) == 0:
        return {f"auc@{t}": float("nan") for t in thresholds}

    errs = _clamp_failures(errors_deg, thresholds)
    errs = [0.0] + sorted(errs)  # start at 0
    recall = np.linspace(0.0, 1.0, len(errs)).tolist()

    out = {}
    for thr in thresholds:
        thr = float(thr)
        last = int(np.searchsorted(errs, thr, side="left"))
        x = errs[:last] + [thr]
        y = recall[:last] + [recall[max(last - 1, 0)]]
        key = f"auc@{int(thr) if thr.is_integer() else thr}"
        out[key] = float(np.trapz(y, x) / thr)
    return out


def macc_pose_errors(errors_deg: List[float], thresholds=(5, 10, 20)) -> Dict[str, float]:
    """mAcc@K: fraction of errors < K. Non-finite are treated as failures."""
    thresholds = tuple(float(t) for t in thresholds)
    if len(errors_deg) == 0:
        return {f"macc@{t}": float("nan") for t in thresholds}
    errs = np.asarray(_clamp_failures(errors_deg, thresholds), dtype=np.float64)
    return {f"macc@{t}": float(np.mean(errs < float(t))) for t in thresholds}


# ============================================================
# Track building (non-legacy, sequential linking)
# ============================================================

def build_tracks_from_sequential_pairs(
    pair_matches: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    topk: int = 20000,
    method: str = "default",
    nn_max_d2: float = 50,
) -> Dict[int, dict]:
    tracks: Dict[int, dict] = {}
    next_tid = 0

    prev_point_to_tid: Dict[tuple, int] = {}
    prev_points: Optional[np.ndarray] = None
    prev_tids: Optional[np.ndarray] = None

    for i, (mk0, mk1, mconf) in enumerate(pair_matches):
        if mconf is not None and len(mconf) > 0:
            k = min(int(len(mconf)), topk)
            idx = np.argsort(-mconf)[:k]
            mk0 = mk0[idx]
            mk1 = mk1[idx]
            mconf = mconf[idx]
        else:
            mconf = None

        if method == "nn":
            N_cur = int(mk0.shape[0])
            if N_cur == 0:
                prev_points = None
                prev_tids = None
                continue

            curr_points_list = []
            curr_tids_list = []

            if prev_points is None or prev_points.shape[0] == 0:
                for j in range(N_cur):
                    ptA = mk0[j]
                    ptB = mk1[j]
                    conf_val = float(mconf[j]) if mconf is not None else 1.0

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
                    curr_points_list.append([float(ptB[0]), float(ptB[1])])
                    curr_tids_list.append(tid)

                prev_points = np.asarray(curr_points_list, dtype=np.float32)
                prev_tids = np.asarray(curr_tids_list, dtype=np.int64)
                continue

            used_current = np.zeros(N_cur, dtype=bool)

            for j, prev_pt in enumerate(prev_points):
                diff = mk0 - prev_pt[None, :]
                d2 = np.sum(diff * diff, axis=1)
                k_min = int(np.argmin(d2))

                if d2[k_min] > nn_max_d2:
                    continue
                if used_current[k_min]:
                    continue

                used_current[k_min] = True

                tid = int(prev_tids[j])
                ptB = mk1[k_min]
                conf_val = float(mconf[k_min]) if mconf is not None else 1.0

                tr = tracks[tid]
                tr["points"].append([float(ptB[0]), float(ptB[1])])
                tr["end_id"] = i + 1
                tr["confs"].append(conf_val)

                curr_points_list.append([float(ptB[0]), float(ptB[1])])
                curr_tids_list.append(tid)

            for k_cur in range(N_cur):
                if used_current[k_cur]:
                    continue
                ptA = mk0[k_cur]
                ptB = mk1[k_cur]
                conf_val = float(mconf[k_cur]) if mconf is not None else 1.0

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
                curr_points_list.append([float(ptB[0]), float(ptB[1])])
                curr_tids_list.append(tid)

            prev_points = np.asarray(curr_points_list, dtype=np.float32)
            prev_tids = np.asarray(curr_tids_list, dtype=np.int64)
            continue

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


# ============================================================
# Incremental SfM (bootstrap + PnP + triangulate)
# ============================================================

def KRT_to_P(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.zeros((3, 4), dtype=np.float64)
    P[:, :3] = R
    P[:, 3] = t.reshape(3)
    return K @ P


def triangulate_two_view(
    K0: np.ndarray, R0: np.ndarray, t0: np.ndarray,
    K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
    x0: np.ndarray, x1: np.ndarray
) -> np.ndarray:
    P0 = KRT_to_P(K0, R0, t0)
    P1 = KRT_to_P(K1, R1, t1)
    x0t = x0.T.astype(np.float64)
    x1t = x1.T.astype(np.float64)
    X_h = cv2.triangulatePoints(P0, P1, x0t, x1t)
    X = (X_h[:3] / (X_h[3:4] + 1e-12)).T
    return X.astype(np.float64)


def reproj_errors_px(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    x_proj, _ = cv2.projectPoints(X.astype(np.float64), rvec, t.astype(np.float64), K.astype(np.float64), None)
    x_proj = x_proj.reshape(-1, 2)
    return np.linalg.norm(x_proj - x_obs.astype(np.float64), axis=1)


def estimate_lo_relative_pose_px(x0_px, x1_px, K0, K1, pixel_thr, conf=0.99999):
    import poselib
    from src.utils.warppers import Camera
    camera0 = Camera.from_calibration_matrix(K0).float()
    camera1 = Camera.from_calibration_matrix(K1).float()
    M, info = poselib.estimate_relative_pose(
        x0_px, x1_px,
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {"max_epipolar_error": pixel_thr, "success_prob": conf}
    )

    success = (
        M is not None and
        ((np.asarray(M.t) != np.array([0., 0., 0.])).all() or
         (np.asarray(M.q) != np.array([1., 0., 0., 0.])).all())
    )
    if not success:
        return False, None, None, np.zeros((len(x0_px),), dtype=bool)

    R = np.asarray(M.R, dtype=np.float64)
    t = np.asarray(M.t, dtype=np.float64).reshape(3)
    inliers = np.asarray(info.get("inliers", []), dtype=bool)
    if inliers.size != len(x0_px):
        inliers = np.zeros((len(x0_px),), dtype=bool)

    return True, R, t, inliers


def estimate_lo_absolute_pose_px(x2d_px, X3d, K, pixel_thr, conf=0.99999):
    import poselib
    from src.utils.warppers import Camera

    cam = Camera.from_calibration_matrix(K).float()

    M, info = poselib.estimate_absolute_pose(
        x2d_px, X3d, cam.to_cameradict(),
        {"max_reproj_error": float(pixel_thr), "success_prob": float(conf)},
    )

    success = (
        M is not None and
        ((np.asarray(M.t) != np.array([0., 0., 0.])).all() or
         (np.asarray(M.q) != np.array([1., 0., 0., 0.])).all())
    )
    if not success:
        return False, None, None, np.zeros((len(x2d_px),), dtype=bool)

    R = np.asarray(M.R, dtype=np.float64)
    t = np.asarray(M.t, dtype=np.float64).reshape(3)

    inliers = np.asarray(info.get("inliers", []), dtype=bool)
    if inliers.size != len(x2d_px):
        inliers = np.zeros((len(x2d_px),), dtype=bool)

    return True, R, t, inliers


def online_sfm_from_tracks(
    tracks: Dict[int, dict],
    cams_gt: List[CameraParams],
    bag_size: int,
    pnp_reproj_thr: float = 4.0,
    min_pnp_points: int = 30,
    tri_max_reproj: float = 4.0,
    min_depth_z: float = 1e-6,
    ransac_mode: str = "lo-ransac",
) -> Dict[str, Any]:
    timings = {
        "bootstrap_ms": 0.0,
        "pnp_ms_sum": 0.0,
        "tri_ms_sum": 0.0,
        "total_sfm_ms": 0.0,
    }
    t_sfm0 = time.perf_counter()

    R_est = [None] * bag_size
    t_est = [None] * bag_size
    R_est[0] = np.eye(3, dtype=np.float64)
    t_est[0] = np.zeros((3,), dtype=np.float64)

    K0 = cams_gt[0].K.astype(np.float64)

    obs_by_frame: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(bag_size)]
    for tid, tr in tracks.items():
        s = tr['start_id']
        pts = np.asarray(tr['points'], dtype=np.float64)
        for j in range(len(pts)):
            f = s + j
            if 0 <= f < bag_size:
                obs_by_frame[f].append((tid, pts[j]))

    map_X: Dict[int, np.ndarray] = {}

    t0 = time.perf_counter()

    x0_list, x1_list, tid_list = [], [], []
    for tid, tr in tracks.items():
        if tr['start_id'] == 0 and tr['end_id'] >= 1 and len(tr['points']) >= 2:
            pts = np.asarray(tr['points'], dtype=np.float64)
            x0_list.append(pts[0])
            x1_list.append(pts[1])
            tid_list.append(tid)

    x0 = np.asarray(x0_list, dtype=np.float64)
    x1 = np.asarray(x1_list, dtype=np.float64)

    boot = {
        "success": False,
        "num_corr": int(len(x0)),
        "num_inliers": 0,
    }

    Z0 = np.asarray([], dtype=np.float64)
    Z1 = np.asarray([], dtype=np.float64)

    if len(x0) >= 50:
        K1 = cams_gt[1].K.astype(np.float64)

        if ransac_mode == "lo-ransac":
            ok, R01, t01, inl = estimate_lo_relative_pose_px(
                x0, x1, K0, K1,
                pixel_thr=0.5,
                conf=0.99999
            )
            if ok and int(inl.sum()) >= 30:
                boot["success"] = True
                boot["num_inliers"] = int(inl.sum())

                R_est[1] = R01
                t_est[1] = t01

                x0_in_px2 = x0[inl]
                x1_in_px2 = x1[inl]
                tid_in2 = [tid_list[i] for i, f in enumerate(inl) if f]

                X01 = triangulate_two_view(K0, R_est[0], t_est[0], K1, R_est[1], t_est[1], x0_in_px2, x1_in_px2)

                err0 = reproj_errors_px(K0, R_est[0], t_est[0], X01, x0_in_px2)
                err1 = reproj_errors_px(K1, R_est[1], t_est[1], X01, x1_in_px2)
                Z0 = (R_est[0] @ X01.T + t_est[0].reshape(3, 1))[2]
                Z1 = (R_est[1] @ X01.T + t_est[1].reshape(3, 1))[2]

                good = (err0 < tri_max_reproj) & (err1 < tri_max_reproj) & (Z0 > min_depth_z) & (Z1 > min_depth_z)

                for ii, g in enumerate(good):
                    if not g:
                        continue
                    map_X[tid_in2[ii]] = X01[ii].astype(np.float64)
        else:
            x0n = cv2.undistortPoints(x0.reshape(-1, 1, 2), K0, None).reshape(-1, 2)
            x1n = cv2.undistortPoints(x1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)

            E, inl = cv2.findEssentialMat(
                x0n, x1n, np.eye(3),
                method=cv2.RANSAC, prob=0.999,
                threshold=0.5 / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
            )

            if E is not None and inl is not None:
                inl = inl.reshape(-1).astype(bool)
                x0_in_n = x0n[inl]
                x1_in_n = x1n[inl]
                x0_in_px = x0[inl]
                x1_in_px = x1[inl]
                tid_in = [tid_list[i] for i, f in enumerate(inl) if f]

                if len(x0_in_n) >= 30:
                    _, R01, t01, mask2 = cv2.recoverPose(E, x0_in_n, x1_in_n)

                    if mask2 is not None:
                        mask2 = mask2.reshape(-1).astype(bool)
                    else:
                        mask2 = np.ones((len(x0_in_n),), dtype=bool)

                    x0_in_n2 = x0_in_n[mask2]
                    x1_in_n2 = x1_in_n[mask2]
                    x0_in_px2 = x0_in_px[mask2]
                    x1_in_px2 = x1_in_px[mask2]
                    tid_in2 = [tid_in[i] for i, m in enumerate(mask2) if m]

                    if len(x0_in_n2) >= 30:
                        boot["success"] = True
                        boot["num_inliers"] = int(len(x0_in_n2))

                        R_est[1] = R01.astype(np.float64)
                        t_est[1] = t01.reshape(3).astype(np.float64)

                        X01 = triangulate_two_view(
                            K0, R_est[0], t_est[0],
                            K1, R_est[1], t_est[1],
                            x0_in_px2, x1_in_px2
                        )

                        err0 = reproj_errors_px(K0, R_est[0], t_est[0], X01, x0_in_px2)
                        err1 = reproj_errors_px(K1, R_est[1], t_est[1], X01, x1_in_px2)
                        Z0 = (R_est[0] @ X01.T + t_est[0].reshape(3, 1))[2]
                        Z1 = (R_est[1] @ X01.T + t_est[1].reshape(3, 1))[2]

                        good = (err0 < tri_max_reproj) & (err1 < tri_max_reproj) & (Z0 > min_depth_z) & (Z1 > min_depth_z)

                        for ii, g in enumerate(good):
                            if not g:
                                continue
                            map_X[tid_in2[ii]] = X01[ii].astype(np.float64)

    if boot["success"]:
        print("[boot] Z0>0:", float(np.mean(Z0 > 0)), " Z1>0:", float(np.mean(Z1 > 0)),
              " seed_points:", int(len(map_X)), "/", int(boot["num_inliers"]))
    else:
        print("[boot] failed. corr:", boot["num_corr"], "inliers:", boot["num_inliers"])

    timings["bootstrap_ms"] = (time.perf_counter() - t0) * 1000.0

    if not boot["success"]:
        timings["total_sfm_ms"] = (time.perf_counter() - t_sfm0) * 1000.0
        return {
            "success": False,
            "bootstrap": boot,
            "poses_est": None,
            "per_frame": [],
            "pose_errors_0k": [],
            "rpe": [],
            "drift": {"gt_path_len_m": 0.0, "rot_deg_per_m": float("nan"), "t_dir_deg_per_m": float("nan")},
            "tracking": {
                "bootstrap_success": False,
                "pnp_success_rate": float("nan"),
                "pnp_success_count": 0,
                "pnp_total": 0,
                "longest_success_streak_frames": 0,
                "final_pose_valid": False,
            },
            "final_0N": None,
            "num_map_points": int(len(map_X)),
            "timings": timings,
        }

    per_frame_stats = []
    for k in range(2, bag_size):
        Kk = cams_gt[k].K.astype(np.float64)

        x2d = []
        X3d = []
        for tid, xy in obs_by_frame[k]:
            if tid in map_X:
                x2d.append(xy)
                X3d.append(map_X[tid])

        x2d = np.asarray(x2d, dtype=np.float64)
        X3d = np.asarray(X3d, dtype=np.float64)

        step = {
            "frame": k,
            "pnp_success": False,
            "num_2d3d": int(len(x2d)),
            "num_inliers": 0,
            "median_reproj_inliers": float("nan"),
            "num_new_points": 0,
            "num_map_points": 0,
        }

        if len(x2d) < min_pnp_points:
            step["num_map_points"] = int(len(map_X))
            per_frame_stats.append(step)
            continue

        t_pnp0 = time.perf_counter()

        if ransac_mode == "lo-ransac":
            ok, Rk, tk, inl_mask = estimate_lo_absolute_pose_px(
                x2d, X3d, Kk,
                pixel_thr=pnp_reproj_thr,
                conf=0.99999
            )
            timings["pnp_ms_sum"] += (time.perf_counter() - t_pnp0) * 1000.0

            if (not ok) or (int(inl_mask.sum()) < min_pnp_points):
                step["num_map_points"] = int(len(map_X))
                per_frame_stats.append(step)
                continue

            R_est[k] = Rk
            t_est[k] = tk
            step["pnp_success"] = True
            step["num_inliers"] = int(inl_mask.sum())

            err_inl = reproj_errors_px(Kk, R_est[k], t_est[k], X3d[inl_mask], x2d[inl_mask])

        else:
            ok, rvec, tvec, inl_idx = cv2.solvePnPRansac(
                objectPoints=X3d,
                imagePoints=x2d,
                cameraMatrix=Kk,
                distCoeffs=None,
                iterationsCount=2000,
                reprojectionError=pnp_reproj_thr,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            timings["pnp_ms_sum"] += (time.perf_counter() - t_pnp0) * 1000.0

            if (not ok) or (inl_idx is None) or (len(inl_idx) < min_pnp_points):
                step["num_map_points"] = int(len(map_X))
                per_frame_stats.append(step)
                continue

            inl_idx = inl_idx.reshape(-1)
            Rk, _ = cv2.Rodrigues(rvec)
            R_est[k] = Rk.astype(np.float64)
            t_est[k] = tvec.reshape(3).astype(np.float64)

            step["pnp_success"] = True
            step["num_inliers"] = int(len(inl_idx))

            err_inl = reproj_errors_px(Kk, R_est[k], t_est[k], X3d[inl_idx], x2d[inl_idx])

        step["median_reproj_inliers"] = float(np.median(err_inl)) if len(err_inl) else float("nan")
        t_tri0 = time.perf_counter()

        if R_est[k-1] is not None and t_est[k-1] is not None:
            obs_prev = {tid: xy for tid, xy in obs_by_frame[k-1]}
            obs_now = {tid: xy for tid, xy in obs_by_frame[k]}

            common_tids = [tid for tid in obs_now.keys() if tid in obs_prev and tid not in map_X]
            if len(common_tids) > 0:
                K_prev = cams_gt[k-1].K.astype(np.float64)
                x_prev = np.asarray([obs_prev[tid] for tid in common_tids], dtype=np.float64)
                x_now = np.asarray([obs_now[tid] for tid in common_tids], dtype=np.float64)

                X_new = triangulate_two_view(
                    K_prev, R_est[k-1], t_est[k-1],
                    Kk, R_est[k], t_est[k],
                    x_prev, x_now
                )

                errp = reproj_errors_px(K_prev, R_est[k-1], t_est[k-1], X_new, x_prev)
                errn = reproj_errors_px(Kk, R_est[k], t_est[k], X_new, x_now)

                Zp = (R_est[k-1] @ X_new.T + t_est[k-1].reshape(3, 1))[2]
                Zn = (R_est[k] @ X_new.T + t_est[k].reshape(3, 1))[2]
                good = (errp < tri_max_reproj) & (errn < tri_max_reproj) & (Zp > min_depth_z) & (Zn > min_depth_z)

                num_added = 0
                for ii, g in enumerate(good):
                    if not g:
                        continue
                    map_X[common_tids[ii]] = X_new[ii].astype(np.float64)
                    num_added += 1
                step["num_new_points"] = int(num_added)

        timings["tri_ms_sum"] += (time.perf_counter() - t_tri0) * 1000.0

        step["num_map_points"] = int(len(map_X))
        per_frame_stats.append(step)

    pose_errs = []
    for k in range(1, bag_size):
        if R_est[k] is None:
            pose_errs.append({"frame": k, "valid": False})
            continue

        R_est_0k = R_est[k]
        t_est_0k = t_est[k]
        R_gt_0k, t_gt_0k = _relative_pose_gt(cams_gt[0], cams_gt[k])

        pose_errs.append({
            "frame": k,
            "valid": True,
            "R_err_deg": _rot_err_deg(R_gt_0k, R_est_0k),
            "t_dir_err_deg": _t_dir_err_deg(t_gt_0k, t_est_0k),
        })

    final = pose_errs[-1] if pose_errs else None

    rpe = []
    gt_path_len = 0.0
    for k in range(1, bag_size):
        R_gt_rel, t_gt_rel = _relative_pose_gt(cams_gt[k-1], cams_gt[k])
        gt_path_len += _norm(t_gt_rel)

        if R_est[k] is None or R_est[k-1] is None:
            rpe.append({"edge": f"{k-1}->{k}", "valid": False})
            continue

        R_est_rel, t_est_rel = _relative_pose_from_w2c(R_est[k-1], t_est[k-1], R_est[k], t_est[k])
        rpe.append({
            "edge": f"{k-1}->{k}",
            "valid": True,
            "R_err_deg": _rot_err_deg(R_gt_rel, R_est_rel),
            "t_dir_err_deg": _t_dir_err_deg(t_gt_rel, t_est_rel),
        })

    drift = {
        "gt_path_len_m": float(gt_path_len),
        "rot_deg_per_m": float("nan"),
        "t_dir_deg_per_m": float("nan"),
    }
    if gt_path_len > 1e-9 and final is not None and final.get("valid", False):
        drift["rot_deg_per_m"] = float(final["R_err_deg"] / gt_path_len)
        drift["t_dir_deg_per_m"] = float(final["t_dir_err_deg"] / gt_path_len)

    pnp_success_flags = [st["pnp_success"] for st in per_frame_stats]
    pnp_success_count = int(sum(bool(x) for x in pnp_success_flags))
    pnp_total = int(len(pnp_success_flags))

    frame_success = [bool(boot["success"])] + [bool(x) for x in pnp_success_flags]
    best = cur = 0
    for ok in frame_success:
        cur = (cur + 1) if ok else 0
        best = max(best, cur)

    tracking = {
        "bootstrap_success": bool(boot["success"]),
        "pnp_success_rate": (pnp_success_count / pnp_total) if pnp_total > 0 else float("nan"),
        "pnp_success_count": pnp_success_count,
        "pnp_total": pnp_total,
        "longest_success_streak_frames": int(best),
        "final_pose_valid": bool(final is not None and final.get("valid", False)),
    }

    timings["total_sfm_ms"] = (time.perf_counter() - t_sfm0) * 1000.0

    return {
        "success": True,
        "bootstrap": boot,
        "per_frame": per_frame_stats,
        "pose_errors_0k": pose_errs,
        "rpe": rpe,
        "drift": drift,
        "tracking": tracking,
        "final_0N": final,
        "num_map_points": int(len(map_X)),
        "timings": timings,
    }


# ============================================================
# Matcher wrappers / runners
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if torch.is_tensor(x):
            return float(x.detach().float().cpu().item())
        return float(x)
    except Exception:
        return float(default)


def run_jamma_pair(
    device: torch.device,
    imgA: Path,
    imgB: Path,
    prev_data: Any = None,
    image_idA: int = 0,
    image_idB: int = 1,
    jamma: Optional[PL_JamMa] = None,
):
    if jamma is None:
        raise ValueError("jamma model must be provided to run_jamma_pair")

    image0, s0, m0, p0, *_ = read_megadepth_color(str(imgA), 832, 8, True)
    image1, s1, m1, p1, *_ = read_megadepth_color(str(imgB), 832, 8, True)

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
        image_idA=image_idA,
        image_idB=image_idB,
    )
    if prev_data is not None:
        data['prev_data'] = prev_data

    jamma = jamma.to(device).eval()
    with torch.no_grad():
        result, flops, runtime = jamma(data)

    if 'prev_data' in data:
        del data['prev_data']

    mk0 = result['mkpts0_f_origin']
    mk1 = result['mkpts1_f_origin']
    mconf = result.get('mconf_f', None)

    runtime_ms = None
    if isinstance(runtime, dict):
        for k in ["runtime_ms", "time_ms", "total_ms", "total_time_ms"]:
            if k in runtime:
                runtime_ms = _safe_float(runtime[k], None)
                break
    if runtime_ms is None:
        runtime_ms = _safe_float(runtime, 0.0)

    flops_f = _safe_float(flops, 0.0)

    return mk0, mk1, mconf, flops_f, runtime_ms, result


# ============================================================
# Model builders
# ============================================================

def build_jamma_model(ckpt: Path, device: torch.device) -> PL_JamMa:
    if not ckpt.exists():
        raise FileNotFoundError(f"JamMa ckpt not found: {ckpt}")
    model = PL_JamMa.load_from_checkpoint(str(ckpt), strict=False)
    model = model.to(device).eval()
    return model


def save_final_summary_txt(
    out_root: Path,
    method: str,
    dataset_name: str,
    summary_text: str,
):
    save_dir = out_root / method / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    txt_path = save_dir / "final_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


def filter_full_length_tracks(tracks: Dict[int, dict], bag_size: int) -> Dict[int, dict]:
    out = {}
    for tid, tr in tracks.items():
        start_id = int(tr["start_id"])
        end_id = int(tr["end_id"])
        pts = tr["points"]

        if start_id == 0 and end_id == bag_size - 1 and len(pts) == bag_size:
            out[tid] = tr
    return out


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    bag_files = sorted(args.subset_dir.glob(f"{args.bag_size}bag_*.txt"))
    if not bag_files:
        raise FileNotFoundError(f"No bag files in {args.subset_dir}")

    run_pair_fn: Callable[..., Any]
    run_pair_kwargs: Dict[str, Any] = {"device": device}

    if args.method == "det-jamma":
        config = get_cfg_defaults()
        config.merge_from_file(args.main_cfg_path)
        config.merge_from_file(args.data_cfg_path)

        config.JAMMA.DET.SEARCH_RADIUS = 832 * 2**0.5
        config.JAMMA.DET.FINE_THR = 0.0
        config.JAMMA.USE_COMPILE = False

        profiler = build_profiler(args.profiler_name)

        jamma = PL_JamMa(config, pretrained_ckpt=args.jamma_ckpt, profiler=profiler)
        run_pair_fn = run_jamma_pair
        run_pair_kwargs.update({"jamma": jamma})
        print(f"[matcher] JamMa ckpt={args.jamma_ckpt}")

    elif args.method == "nn-jamma":
        config = get_cfg_defaults()
        config.merge_from_file(args.main_cfg_path)
        config.merge_from_file(args.data_cfg_path)

        config.JAMMA.DET.SEARCH_RADIUS = 832 * 2**0.5
        config.JAMMA.DET.FINE_THR = 0.0
        config.JAMMA.USE_COMPILE = False
        config.JAMMA.DET.USE_DET = False

        profiler = build_profiler(args.profiler_name)

        jamma = PL_JamMa(config, pretrained_ckpt=args.jamma_ckpt, profiler=profiler)
        run_pair_fn = run_jamma_pair
        run_pair_kwargs.update({"jamma": jamma})
        print(f"[matcher] NN-JamMa ckpt={args.jamma_ckpt}")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    all_bag_results = []

    total_match_time_ms = 0.0
    total_match_flops = 0.0
    total_sfm_time_ms = 0.0
    total_pairs = 0
    total_num_map_points = 0.0

    final_errors_deg: List[float] = []
    rpe_errors_deg: List[float] = []
    prefix_errors_deg: List[float] = []

    for bf in bag_files:
        rel_paths = read_bag_paths(bf)
        img_paths = [args.dataset_root / rp for rp in rel_paths]
        cams = [load_cam_from_dir(args.calib_dir, p, args.flip_w2c) for p in img_paths]

        prev_data = None
        pair_matches = []
        bag_match_time_ms = 0.0
        bag_match_flops = 0.0
        image_idA = 0
        image_idB = 1

        for i in range(args.bag_size - 1):
            mk0_t, mk1_t, mconf_t, flops, runtime_ms, prev_result = run_pair_fn(
                imgA=img_paths[i],
                imgB=img_paths[i + 1],
                prev_data=prev_data,
                image_idA=image_idA,
                image_idB=image_idB,
                **run_pair_kwargs,
            )

            prev_data = prev_result
            image_idA += 1
            image_idB += 1

            bag_match_time_ms += float(runtime_ms)
            bag_match_flops += float(flops)

            mk0 = mk0_t.detach().cpu().numpy() if hasattr(mk0_t, "detach") else np.asarray(mk0_t)
            mk1 = mk1_t.detach().cpu().numpy() if hasattr(mk1_t, "detach") else np.asarray(mk1_t)
            mconf = mconf_t.detach().cpu().numpy() if (mconf_t is not None and hasattr(mconf_t, "detach")) else (
                None if mconf_t is None else np.asarray(mconf_t)
            )
            pair_matches.append((mk0, mk1, mconf))

        track_method = "nn" if args.method == "nn-jamma" else "default"
        tracks = build_tracks_from_sequential_pairs(
            pair_matches,
            topk=args.topk,
            method=track_method,
            nn_max_d2=50
        )

        lengths = [len(tr["points"]) for tr in tracks.values()]
        print("[tracks] num:", len(lengths),
              "len>=2:", sum(l >= 2 for l in lengths),
              "len>=3:", sum(l >= 3 for l in lengths),
              "len>=4:", sum(l >= 4 for l in lengths),
              "len>=5:", sum(l >= 5 for l in lengths),
              "maxlen:", max(lengths) if lengths else 0)

        tracks = filter_full_length_tracks(tracks, bag_size=args.bag_size)

        sfm_res = online_sfm_from_tracks(
            tracks,
            cams_gt=cams,
            bag_size=args.bag_size,
            pnp_reproj_thr=args.pnp_reproj_thr,
            min_pnp_points=args.min_pnp_points,
            tri_max_reproj=args.tri_max_reproj,
            min_depth_z=args.min_depth_z,
            ransac_mode=args.ransac_mode,
        )

        num_map_points = int(sfm_res.get("num_map_points", 0))
        total_num_map_points += num_map_points

        total_pairs += (args.bag_size - 1)
        total_match_time_ms += bag_match_time_ms
        total_match_flops += bag_match_flops
        total_sfm_time_ms += float(sfm_res.get("timings", {}).get("total_sfm_ms", 0.0))

        if not sfm_res.get("success", False):
            final_errors_deg.append(float("inf"))
            for _ in range(args.bag_size - 1):
                rpe_errors_deg.append(float("inf"))
            for _ in range(1, args.bag_size):
                prefix_errors_deg.append(float("inf"))
        else:
            fin = sfm_res.get("final_0N", None)
            if fin is not None and fin.get("valid", False):
                final_errors_deg.append(_pose_max_error_deg(fin["R_err_deg"], fin["t_dir_err_deg"]))
            else:
                final_errors_deg.append(float("inf"))

            for e in sfm_res.get("rpe", []):
                if e.get("valid", False):
                    rpe_errors_deg.append(_pose_max_error_deg(e["R_err_deg"], e["t_dir_err_deg"]))
                else:
                    rpe_errors_deg.append(float("inf"))

            for pe in sfm_res.get("pose_errors_0k", []):
                if pe.get("valid", False):
                    prefix_errors_deg.append(_pose_max_error_deg(pe["R_err_deg"], pe["t_dir_err_deg"]))
                else:
                    prefix_errors_deg.append(float("inf"))

        print(f"\n[Bag] {bf.name}")
        print(f"  matcher: time_sum={bag_match_time_ms:.2f} ms, flops_sum={bag_match_flops:.3e} "
              f"(avg_time/pair={bag_match_time_ms / max(1, args.bag_size - 1):.2f} ms, "
              f"avg_flops/pair={bag_match_flops / max(1, args.bag_size - 1):.3e})")
        print(f"  sfm: success={sfm_res['success']} map_points={sfm_res.get('num_map_points', 0)} "
              f"timings={sfm_res.get('timings', {})}")
        print("  bootstrap:", sfm_res.get("bootstrap", {}))
        print("  tracking:", sfm_res.get("tracking", {}))
        print("  drift:", sfm_res.get("drift", {}))

        if sfm_res["success"]:
            print("  pose errors (0->k):")
            for pe in sfm_res.get("pose_errors_0k", []):
                if not pe.get("valid", False):
                    print(f"    k={pe['frame']}: invalid")
                else:
                    print(f"    k={pe['frame']}: R_err={pe['R_err_deg']:.2f}deg, t_dir_err={pe['t_dir_err_deg']:.2f}deg")

            print("  RPE (k-1->k):")
            for e in sfm_res.get("rpe", []):
                if not e.get("valid", False):
                    print(f"    {e['edge']}: invalid")
                else:
                    print(f"    {e['edge']}: R_err={e['R_err_deg']:.2f}deg, t_dir_err={e['t_dir_err_deg']:.2f}deg")

            print("  per-frame incremental stats:")
            for st in sfm_res.get("per_frame", []):
                print(f"    k={st['frame']}: 2D3D={st['num_2d3d']} inl={st['num_inliers']} "
                      f"med_reproj={st['median_reproj_inliers']:.2f}px newX={st['num_new_points']} "
                      f"mapX={st['num_map_points']} pnp_ok={st['pnp_success']}")

        all_bag_results.append({
            "bag": str(bf),
            "num_tracks": int(len(tracks)),
            "matcher": {
                "time_ms_sum": float(bag_match_time_ms),
                "flops_sum": float(bag_match_flops),
                "time_ms_avg_per_pair": float(bag_match_time_ms / max(1, args.bag_size - 1)),
                "flops_avg_per_pair": float(bag_match_flops / max(1, args.bag_size - 1)),
                "pairs": int(args.bag_size - 1),
            },
            "sfm": sfm_res,
        })

    dataset_name_for_save = "phototourism" if use_dataset == "IMC" else "megadepth"

    summary_lines = []
    summary_lines.append("=== Global Summary ===")
    if total_pairs > 0:
        summary_lines.append(f"pairs_total={total_pairs}")
        summary_lines.append(f"matcher_avg_time_per_pair(ms)={total_match_time_ms / total_pairs:.4f}")
        summary_lines.append(f"matcher_avg_FLOPs_per_pair={total_match_flops / total_pairs:.4e}")
        summary_lines.append(f"sfm_avg_time_per_bag(ms)={total_sfm_time_ms / max(1, len(bag_files)):.4f}")
        summary_lines.append(f"avg_num_3d_points_per_bag={total_num_map_points / max(1, len(bag_files)):.4f}")
    else:
        summary_lines.append("No pairs processed.")

    thr = tuple(float(x) for x in args.auc_thresholds)

    auc_final = auc_pose_errors(final_errors_deg, thresholds=thr)
    auc_rpe = auc_pose_errors(rpe_errors_deg, thresholds=thr)
    auc_prefix = auc_pose_errors(prefix_errors_deg, thresholds=thr)

    macc_final = macc_pose_errors(final_errors_deg, thresholds=thr)
    macc_rpe = macc_pose_errors(rpe_errors_deg, thresholds=thr)
    macc_prefix = macc_pose_errors(prefix_errors_deg, thresholds=thr)

    def _fmt_auc(d: Dict[str, float]) -> str:
        parts = []
        for k in sorted(d.keys(), key=lambda s: float(s.split("@")[1])):
            v = d[k]
            parts.append(f"{k.upper()}={v * 100:.2f}" if np.isfinite(v) else f"{k.upper()}=nan")
        return "  ".join(parts)

    def _fmt_macc(d: Dict[str, float]) -> str:
        parts = []
        for k in sorted(d.keys(), key=lambda s: float(s.split("@")[1])):
            v = d[k]
            parts.append(f"{k.replace('macc', 'mAcc')}={v * 100:.2f}" if np.isfinite(v) else f"{k.replace('macc', 'mAcc')}=nan")
        return "  ".join(parts)

    global scene
    summary_lines.append("")
    summary_lines.append("=== AUC Metrics (IMC-like; error=max(rot, tdir) deg; FAILURES INCLUDED) ===")
    summary_lines.append(f"method {args.method}, scene {scene}")
    summary_lines.append(f"[Multiview-like] final 0->N :  {_fmt_auc(auc_final)}")
    summary_lines.append(f"                 {_fmt_macc(macc_final)}")
    summary_lines.append(f"[Stereo-like]    RPE edges  :  {_fmt_auc(auc_rpe)}")
    summary_lines.append(f"                 {_fmt_macc(macc_rpe)}")
    summary_lines.append(f"[Online-like]    prefix 0->k:  {_fmt_auc(auc_prefix)}")
    summary_lines.append(f"                 {_fmt_macc(macc_prefix)}")
    summary_lines.append(f"Counts: final={len(final_errors_deg)}  rpe={len(rpe_errors_deg)}  prefix={len(prefix_errors_deg)}")

    summary_text = "\n".join(summary_lines)

    print()
    print(summary_text)

    save_final_summary_txt(
        out_root=Path("./sfm_txt_results"),
        method=args.method,
        dataset_name=dataset_name_for_save,
        summary_text=summary_text,
    )

    summary = {
        "pairs_total": int(total_pairs),
        "num_bags": int(len(bag_files)),
        "matcher_avg_time_per_pair_ms": float(total_match_time_ms / total_pairs) if total_pairs > 0 else float("nan"),
        "matcher_avg_flops_per_pair": float(total_match_flops / total_pairs) if total_pairs > 0 else float("nan"),
        "sfm_avg_time_per_bag_ms": float(total_sfm_time_ms / max(1, len(bag_files))),
        "avg_num_3d_points_per_bag": float(total_num_map_points / max(1, len(bag_files))),
        "auc_thresholds_deg": [float(x) for x in thr],
        "auc": {
            "final_0N": {k: float(v) for k, v in auc_final.items()},
            "rpe": {k: float(v) for k, v in auc_rpe.items()},
            "prefix_0k": {k: float(v) for k, v in auc_prefix.items()},
            "macc_final_0N": {k: float(v) for k, v in macc_final.items()},
            "macc_rpe": {k: float(v) for k, v in macc_rpe.items()},
            "macc_prefix_0k": {k: float(v) for k, v in macc_prefix.items()},
            "counts": {
                "num_final": int(len(final_errors_deg)),
                "num_rpe": int(len(rpe_errors_deg)),
                "num_prefix": int(len(prefix_errors_deg)),
            },
        },
    }

    if args.out_json is not None:
        payload = {"summary": summary, "bags": all_bag_results}
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved: {args.out_json}")


use_dataset = 'megadepth'  # IMC or megadepth

if use_dataset == 'IMC':
    default_root = Path('/home/ach17765lb/data/phototourism')
    scene = 'st_peters_square'
    set_name = 'set_100'
    subset_size = 10
    subset_dir = default_root / scene / set_name / 'sub_set'
    dataset_root = default_root / scene / set_name
    calib_dir = default_root / scene / set_name / 'calibration'
else:
    default_root = Path('/home/ach17765lb/JamMa/data/megadepth/Undistorted_SfM/')
    scene = '0015'
    subset_size = 10
    subset_dir = default_root / scene / '5bag'
    dataset_root = Path('/home/ach17765lb/JamMa/data/megadepth/')
    calib_dir = default_root / scene / 'calibration'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset_dir', type=Path, default=subset_dir,
                    help='Contains Nbag_*.txt files (e.g., 5bag_015.txt)')
    ap.add_argument('--dataset_root', type=Path, default=dataset_root,
                    help='Root to prepend to image relative paths in bag files')
    ap.add_argument('--calib_dir', type=Path, default=calib_dir,
                    help='Directory with calibration_<stem>.h5 per image')
    ap.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_test_1500.py")
    ap.add_argument('--main_cfg_path', type=str, default="configs/jamma/outdoor/test.py")
    ap.add_argument('--profiler_name', type=str, default="inference")

    ap.add_argument("--bag_size", type=int, default=5)
    ap.add_argument("--flip_w2c", action="store_true")
    ap.add_argument("--topk", type=int, default=20000)
    ap.add_argument("--method", type=str, default="nn-jamma", choices=["nn-jamma", "det-jamma"])
    ap.add_argument("--ransac_mode", type=str, default="lo-ransac", choices=["ransac", "lo-ransac"])
    ap.add_argument("--out_json", type=Path, default=None)

    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--pnp_reproj_thr", type=float, default=4.0)
    ap.add_argument("--min_pnp_points", type=int, default=30)
    ap.add_argument("--tri_max_reproj", type=float, default=4.0)
    ap.add_argument("--min_depth_z", type=float, default=1e-6)

    ap.add_argument("--jamma_ckpt", type=str, default="official")

    ap.add_argument("--auc_thresholds", type=float, nargs="+", default=[5.0, 10.0],
                    help="AUC/mAcc thresholds in degrees, e.g. --auc_thresholds 5 10 20")

    return ap.parse_args()


if __name__ == "__main__":
    main()