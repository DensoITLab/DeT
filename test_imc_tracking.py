#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Any, Optional

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import cv2
import pytorch_lightning as pl

# --- JamMa imports (project-local) ---
from src.utils.dataset import read_megadepth_color
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler

from src.utils.metrics import (
    estimate_lo_pose,
    relative_pose_error,
    symmetric_epipolar_distance,
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
)

from collections import OrderedDict


# --- Dataset defaults (PhotoTourism example) ---
default_root = Path('data/imc')
scene = 'st_peters_square'   # 'reichstag' , 'sacre_coeur', 'st_peters_square'
set_name = 'set_100'
subset_size = 5


# ============================================================
# 1) Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Config paths (JamMa setup) ---
    parser.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument('--main_cfg_path', type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument('--ckpt_path', type=str, default="official")
    parser.add_argument('--dump_dir', type=str, default="dump/jamma_outdoor")
    parser.add_argument('--profiler_name', type=str, default="inference")
    parser.add_argument('--thr', type=float, default=None)


    parser.add_argument('--subset_dir', type=Path, default=default_root / scene / set_name / 'sub_set',
                        help='Contains Nbag_*.txt files (e.g., 5bag_015.txt)')
    parser.add_argument('--dataset_root', type=Path, default=default_root / scene / set_name,
                        help='Root to prepend to image relative paths in bag files')
    parser.add_argument('--calib_dir', type=Path, default=default_root / scene / set_name / 'calibration',
                        help='Directory with calibration_<stem>.h5 per image')
    parser.add_argument('--depth_dir', type=Path, default=default_root / scene / set_name / 'depth_maps',
                        help='Directory with <stem>.h5 per image (depth)')

    # --- pose consistency thresholds (0->N) ---
    parser.add_argument('--min_corr_0N', type=int, default=30,
                        help='Minimum correspondences to attempt 0->N pose estimation')
    parser.add_argument('--min_inliers_0N', type=int, default=30,
                        help='Minimum inliers for accepting 0->N pose estimation as valid')
    parser.add_argument('--max_pose_err_0N_deg', type=float, default=5.0,
                        help='Max pose error (max(R_err, t_err)) to call bag pose-consistent')

    parser.add_argument('--bag_size', type=int, default=subset_size, help='Number of images in each bag file')
    parser.add_argument('--flip_w2c', action='store_true',
                        help='If calib is world->cam, convert to cam->world internally')

    parser.add_argument('--save_json', type=Path, default=Path('outputs/fig6_imc_tracking.json'),
                        help='Full results (per_bag + summary)')
    parser.add_argument('--save_summary_json', type=Path,
                        default=Path('outputs/fig6_imc_tracking_summary.json'),
                        help='Only global summary dict per method')
    parser.add_argument('--dataset_name', type=str, default='imc',
                        help='Dataset label stored in output JSON')
    parser.add_argument('--scene_name', type=str, default=scene,
                        help='Scene label stored in output JSON')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['jamma'],
        help=(
            'Which methods to run. '
            'Subset of {jamma, det-jamma, jamma_legacy}. '
            'Example: --methods det-jamma'
        ),
    )

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ============================================================
# 2) Calibration utilities
# ============================================================

@dataclasses.dataclass
class CameraParams:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


def _read_cam_from_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load K, R, T (or t) from calibration file."""
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
    """Read per-image calibration and optionally flip world-to-camera."""
    h5_path = calib_dir / f"calibration_{img_path.stem}.h5"
    K, R, t = _read_cam_from_h5(h5_path)
    if flip_w2c:
        R, t = R.T, -R.T @ t
    return CameraParams(K, R, t)


# ============================================================
# 2') Depth utilities
# ============================================================

def _read_depth_from_h5(h5_path: Path) -> np.ndarray:
    """Load depth map from h5 file. Shape: (H, W)."""
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, 'r') as f:
        if 'depth' in f:
            depth = np.array(f['depth'])
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


# ============================================================
# 3) Geometry / metrics
# ============================================================

def _relative_pose(camA: CameraParams, camB: CameraParams):
    """Relative pose of camB wrt camA."""
    R21 = camB.R @ camA.R.T
    t21 = camB.t - R21 @ camA.t
    return R21, t21


def _build_T_0to1(camA: CameraParams, camB: CameraParams) -> np.ndarray:
    R21, t21 = _relative_pose(camA, camB)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R21
    T[:3, 3] = t21
    return T


def _compute_pair_metrics_jamma_style(
    mk0: np.ndarray,
    mk1: np.ndarray,
    camA: CameraParams,
    camB: CameraParams,
    config,
    device: torch.device,
    epi_thr: float = 1e-4,
) -> dict:
    n_total = int(mk0.shape[0])
    if n_total == 0:
        return dict(
            precision=0.0,
            mean_err=float('nan'),
            median_err=float('nan'),
            n_correct=0,
            n_total=0,
            R_err_deg=float('nan'),
            t_err_deg=float('nan'),
            epi_errs=np.zeros((0,), dtype=float),
        )

    T_0to1_np = _build_T_0to1(camA, camB)

    data = {
        'mkpts0_f_origin': torch.from_numpy(mk0).float().to(device),
        'mkpts1_f_origin': torch.from_numpy(mk1).float().to(device),
        'K0': torch.from_numpy(camA.K).float().unsqueeze(0).to(device),
        'K1': torch.from_numpy(camB.K).float().unsqueeze(0).to(device),
        'T_0to1': torch.from_numpy(T_0to1_np).float().unsqueeze(0).to(device),
        'm_bids': torch.zeros(mk0.shape[0], dtype=torch.long, device=device),
    }

    compute_symmetrical_epipolar_errors(data)
    epi_errs = data['epi_errs'].detach().cpu().numpy().astype(np.float64)

    n_total = int(epi_errs.size)
    n_correct = int((epi_errs < epi_thr).sum()) if n_total else 0
    precision = float(n_correct / n_total) if n_total else 0.0
    mean_err = float(epi_errs.mean()) if n_total else float('nan')
    median_err = float(np.median(epi_errs)) if n_total else float('nan')

    compute_pose_errors(data, config)

    if len(data['R_errs']) > 0 and len(data['R_errs'][0]) > 0:
        R_err_deg = float(data['R_errs'][0][0])
        t_err_deg = float(data['t_errs'][0][0])
    else:
        R_err_deg = float('nan')
        t_err_deg = float('nan')

    return dict(
        precision=precision,
        mean_err=mean_err,
        median_err=median_err,
        n_correct=n_correct,
        n_total=n_total,
        R_err_deg=R_err_deg,
        t_err_deg=t_err_deg,
        epi_errs=epi_errs,
    )


def _compute_symmetric_epi_errors_for_two_cams(
    x0_px: np.ndarray,
    x1_px: np.ndarray,
    cam0: CameraParams,
    cam1: CameraParams,
    device: torch.device,
) -> np.ndarray:
    if x0_px.size == 0:
        return np.zeros((0,), dtype=float)

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
# ============================================================

def _empty_reproj_stats(thr_list=None) -> Dict[str, Any]:
    if thr_list is None:
        thr_list = [1, 3, 5, 10]
    return {
        'thr_px': list(thr_list),
        'counts': [0] * len(thr_list),
        'ratios': [0.0] * len(thr_list),
        'n_valid': 0,
        'median_err_px': float('nan'),
    }


def _compute_depth_errors_between_two_views(
    p_src: np.ndarray,
    p_tgt: np.ndarray,
    depth_src: Optional[np.ndarray],
    depth_tgt: Optional[np.ndarray],
    cam_src: CameraParams,
    cam_tgt: CameraParams,
    use_rel_err_gate: bool = True,
) -> Tuple[float, bool, float, bool]:
    # ---------- forward: k -> k+1 ----------
    e_fwd = 0.0
    valid_fwd = False

    if depth_src is not None:
        Hs, Ws = depth_src.shape
        xs, ys = float(p_src[0]), float(p_src[1])
        ix = int(round(xs))
        iy = int(round(ys))

        if 0 <= ix < Ws and 0 <= iy < Hs:
            Z_src = float(depth_src[iy, ix])
            if np.isfinite(Z_src) and Z_src > 0:
                K_s_inv = np.linalg.inv(cam_src.K)
                R_s, t_s = cam_src.R, cam_src.t.reshape(3,)
                R_t, t_t = cam_tgt.R, cam_tgt.t.reshape(3,)

                p_h = np.array([xs, ys, 1.0], dtype=np.float32)
                X_cam_s = K_s_inv @ (p_h * Z_src)
                X_world = R_s.T @ (X_cam_s - t_s)

                X_cam_t = R_t @ X_world + t_t
                if X_cam_t[2] > 0:
                    p_t = cam_tgt.K @ (X_cam_t / X_cam_t[2])
                    u_t, v_t = float(p_t[0]), float(p_t[1])

                    if depth_tgt is not None:
                        Ht, Wt = depth_tgt.shape
                        if 0 <= u_t < Wt and 0 <= v_t < Ht:
                            iu_t = int(round(u_t))
                            iv_t = int(round(v_t))
                            if 0 <= iu_t < Wt and 0 <= iv_t < Ht:
                                if use_rel_err_gate:
                                    Z_tgt = float(depth_tgt[iv_t, iu_t])
                                    if np.isfinite(Z_tgt) and Z_tgt > 0:
                                        Z_pred_t = float(X_cam_t[2])
                                        rel_err = abs(Z_pred_t - Z_tgt) / max(abs(Z_tgt), 1e-8)
                                        if rel_err < 0.2:
                                            e_fwd = float(np.linalg.norm(
                                                p_tgt - np.array([u_t, v_t], dtype=np.float32)
                                            ))
                                            valid_fwd = True
                                else:
                                    e_fwd = float(np.linalg.norm(
                                        p_tgt - np.array([u_t, v_t], dtype=np.float32)
                                    ))
                                    valid_fwd = True
                    else:
                        e_fwd = float(np.linalg.norm(
                            p_tgt - np.array([u_t, v_t], dtype=np.float32)
                        ))
                        valid_fwd = True

    # ---------- backward: k+1 -> k ----------
    e_bwd = 0.0
    valid_bwd = False

    if depth_tgt is not None:
        Ht, Wt = depth_tgt.shape
        xt, yt = float(p_tgt[0]), float(p_tgt[1])
        ix = int(round(xt))
        iy = int(round(yt))

        if 0 <= ix < Wt and 0 <= iy < Ht:
            Z_tgt = float(depth_tgt[iy, ix])
            if np.isfinite(Z_tgt) and Z_tgt > 0:
                K_t_inv = np.linalg.inv(cam_tgt.K)
                R_t, t_t = cam_tgt.R, cam_tgt.t.reshape(3,)
                R_s, t_s = cam_src.R, cam_src.t.reshape(3,)

                p_h = np.array([xt, yt, 1.0], dtype=np.float32)
                X_cam_t = K_t_inv @ (p_h * Z_tgt)
                X_world = R_t.T @ (X_cam_t - t_t)

                X_cam_s = R_s @ X_world + t_s
                if X_cam_s[2] > 0:
                    p_s = cam_src.K @ (X_cam_s / X_cam_s[2])
                    u_s, v_s = float(p_s[0]), float(p_s[1])

                    if depth_src is not None:
                        Hs, Ws = depth_src.shape
                        iu_s_gt = int(round(float(p_src[0])))
                        iv_s_gt = int(round(float(p_src[1])))
                        if (0 <= u_s < Ws and 0 <= v_s < Hs and
                                0 <= iu_s_gt < Ws and 0 <= iv_s_gt < Hs):
                            if use_rel_err_gate:
                                Z_src = float(depth_src[iv_s_gt, iu_s_gt])
                                if np.isfinite(Z_src) and Z_src > 0:
                                    Z_pred_s = float(X_cam_s[2])
                                    rel_err = abs(Z_pred_s - Z_src) / max(abs(Z_src), 1e-8)
                                    if rel_err < 0.2:
                                        e_bwd = float(np.linalg.norm(
                                            p_src - np.array([u_s, v_s], dtype=np.float32)
                                        ))
                                        valid_bwd = True
                            else:
                                e_bwd = float(np.linalg.norm(
                                    p_src - np.array([u_s, v_s], dtype=np.float32)
                                ))
                                valid_bwd = True
                    else:
                        e_bwd = float(np.linalg.norm(
                            p_src - np.array([u_s, v_s], dtype=np.float32)
                        ))
                        valid_bwd = True

    return e_fwd, valid_fwd, e_bwd, valid_bwd


# ============================================================
# 3'') depth-based GT visibility from image0
# ============================================================

def _compute_depth_gt_visibility_from0(
    mk0_01: np.ndarray,
    depth0: np.ndarray,
    cams: List[CameraParams],
    depths: List[Optional[np.ndarray]],
    mutual_backcheck_thr_px: float = 1.0,
) -> Dict[str, Any]:
    num_cams = len(cams)
    if num_cams == 0 or mk0_01.shape[0] == 0:
        return {
            'num_points_initial': 0,
            'counts_per_k': [0] * num_cams,
            'counts_per_k_vis_only': [0] * num_cams,
            'multi_thr': {
                'thr_px': [1.0, 3.0, 5.0, 10.0],
                'counts_per_k': [[0] * num_cams for _ in range(4)],
            },
        }

    thr_list = [1.0, 3.0, 5.0, 10.0]

    depth0 = depth0.astype(np.float32)
    H0, W0 = depth0.shape

    init_px = []
    for (x0, y0) in mk0_01:
        ix = int(round(float(x0)))
        iy = int(round(float(y0)))
        if ix < 0 or ix >= W0 or iy < 0 or iy >= H0:
            continue
        Z = float(depth0[iy, ix])
        if not np.isfinite(Z) or Z <= 0:
            continue
        init_px.append([float(x0), float(y0)])

    if not init_px:
        return {
            'num_points_initial': 0,
            'counts_per_k': [0] * num_cams,
            'counts_per_k_vis_only': [0] * num_cams,
            'multi_thr': {
                'thr_px': thr_list,
                'counts_per_k': [[0] * num_cams for _ in thr_list],
            },
        }

    cur_pix = np.asarray(init_px, dtype=np.float32)
    P = cur_pix.shape[0]

    counts_per_k_vis_only = [0] * num_cams
    alive_vis_only = np.ones(P, dtype=bool)
    counts_per_k_vis_only[0] = int(alive_vis_only.sum())

    counts_per_k_thr = {t: [0] * num_cams for t in thr_list}
    alive_sym = {t: np.ones(P, dtype=bool) for t in thr_list}
    for t in thr_list:
        counts_per_k_thr[t][0] = P

    depths_fixed = list(depths)
    if len(depths_fixed) > 0:
        depths_fixed[0] = depth0
    else:
        depths_fixed = [depth0]

    # --- step-by-step: frame k -> k+1 ---
    for k in range(num_cams - 1):
        cam_k = cams[k]
        cam_n = cams[k + 1]
        depth_k = depths_fixed[k] if k < len(depths_fixed) else None
        depth_n = depths_fixed[k + 1] if (k + 1) < len(depths_fixed) else None

        if depth_k is None:
            alive_vis_only[:] = False
            counts_per_k_vis_only[k + 1] = 0
            for t in thr_list:
                alive_sym[t][:] = False
                counts_per_k_thr[t][k + 1] = 0
            continue

        Hk, Wk = depth_k.shape
        Hn, Wn = (depth_n.shape if depth_n is not None else (None, None))

        old_cur_pix = cur_pix.copy()
        new_pix = cur_pix.copy()
        new_alive = np.zeros_like(alive_vis_only)

        for i in range(P):
            if not alive_vis_only[i]:
                continue

            xk, yk = float(old_cur_pix[i, 0]), float(old_cur_pix[i, 1])
            ix = int(round(xk))
            iy = int(round(yk))

            if ix < 0 or ix >= Wk or iy < 0 or iy >= Hk:
                alive_vis_only[i] = False
                continue

            Z = float(depth_k[iy, ix])
            if not np.isfinite(Z) or Z <= 0:
                alive_vis_only[i] = False
                continue

            Kk_inv = np.linalg.inv(cam_k.K)
            Rk, tk = cam_k.R, cam_k.t.reshape(3,)
            Rn, tn = cam_n.R, cam_n.t.reshape(3,)

            p_h = np.array([xk, yk, 1.0], dtype=np.float32)
            X_cam_k = Kk_inv @ (p_h * Z)
            X_world = Rk.T @ (X_cam_k - tk)

            X_cam_n = Rn @ X_world + tn
            if X_cam_n[2] <= 0:
                alive_vis_only[i] = False
                continue

            p_n = cam_n.K @ (X_cam_n / X_cam_n[2])
            u, v = float(p_n[0]), float(p_n[1])

            if Hn is not None and Wn is not None:
                if u < 0 or u >= Wn or v < 0 or v >= Hn:
                    alive_vis_only[i] = False
                    continue

            new_pix[i, 0] = u
            new_pix[i, 1] = v
            new_alive[i] = True

        cur_pix = new_pix
        alive_vis_only = new_alive
        counts_per_k_vis_only[k + 1] = int(alive_vis_only.sum())

        for i in range(P):
            if not alive_vis_only[i]:
                for t in thr_list:
                    alive_sym[t][i] = False
                continue

            p_k = old_cur_pix[i]
            p_n = cur_pix[i]

            e_fwd, valid_fwd, e_bwd, valid_bwd = _compute_depth_errors_between_two_views(
                p_src=p_k,
                p_tgt=p_n,
                depth_src=depth_k,
                depth_tgt=depth_n,
                cam_src=cam_k,
                cam_tgt=cam_n
            )

            for t in thr_list:
                if not alive_sym[t][i]:
                    continue
                if not valid_bwd:
                    alive_sym[t][i] = False
                else:
                    if e_bwd > t:
                        alive_sym[t][i] = False


        for t in thr_list:
            counts_per_k_thr[t][k + 1] = int(alive_sym[t].sum())

    base_thr = min(thr_list, key=lambda x: abs(x - float(mutual_backcheck_thr_px)))
    counts_per_k_base = counts_per_k_thr[base_thr]

    return {
        'num_points_initial': int(P),
        'counts_per_k': [int(c) for c in counts_per_k_base],
        'counts_per_k_vis_only': [int(c) for c in counts_per_k_vis_only],
        'multi_thr': {
            'thr_px': thr_list,
            'counts_per_k': [
                [int(c) for c in counts_per_k_thr[t]] for t in thr_list
            ],
        },
    }


# ============================================================
# 4) Matcher forward per pair
# ============================================================


def run_jamma_pair(device: torch.device,
                   imgA: Path,
                   imgB: Path,
                   prev_data: Any = None,
                   image_idA: int = 0,
                   image_idB: int = 1,
                   jamma: PL_JamMa = None):
    """Run JamMa on a single image pair and return matches (origin coords)."""
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
    print(runtime)

    return mk0, mk1, mconf, flops, runtime, result




# ============================================================
# 5) Bag evaluation with tracks (and 0→N / 0→k metrics)
# ============================================================

def read_bag_paths(bag_file: Path) -> List[str]:
    return [ln.strip() for ln in bag_file.read_text(encoding='utf-8').splitlines() if ln.strip()]


def evaluate_bag_with_tracks(
    bag_file: Path,
    bag_size: int,
    dataset_root: Path,
    calib_dir: Path,
    flip_w2c: bool,
    run_pair_fn: Callable[..., Any],
    run_pair_kwargs: dict,
    device: torch.device,
    config,
    epi_threshold: float = 1e-4,
    topk: int = 20000,
    legacy_linking: bool = False,
    depth_dir: Path = None,
    compute_depth_gt_from0: bool = False,
    min_corr_0N: int = 30,
    min_inliers_0N: int = 30,
    max_pose_err_0N_deg: float = 5.0,
) -> dict:
    """
    Run pairs 0-1,1-2,...,(N-2)-(N-1) with arbitrary matcher (run_pair_fn),
    link tracks, and compute 0→N / 0→k metrics from tracks.
    """
    rel_paths = read_bag_paths(bag_file)
    if len(rel_paths) != bag_size:
        raise ValueError(f"{bag_file}: expected {bag_size} paths, got {len(rel_paths)}")
    img_paths = [dataset_root / rp for rp in rel_paths]

    cams = [load_cam_from_dir(calib_dir, p, flip_w2c) for p in img_paths]

    if depth_dir is not None:
        depths = []
        for p in img_paths:
            try:
                depths.append(load_depth_from_dir(depth_dir, p))
            except Exception as e:
                print(f"[WARN] depth load failed for {p}: {e}")
                depths.append(None)
    else:
        depths = [None] * len(img_paths)

    pair_results = []
    tracks: Dict[int, dict] = {}
    next_tid = 0

    flops_total = 0.0
    runtime_ms_total = 0.0
    num_pairs = bag_size - 1

    prev_point_to_tid: Dict[tuple, int] = {}
    prev_points = None
    prev_tids = None
    prev_data = None

    pixel_thr = 0.5
    conf = 0.999

    mk0_01_for_depth_gt = None
    image_idA = 0
    image_idB = 1

    for i in range(num_pairs):
        imgA, imgB = img_paths[i], img_paths[i + 1]
        print(f"→ Pair {i}: {imgA.name} ↔ {imgB.name}")

        mk0_t, mk1_t, mconf, flops, runtime_ms, prev_result = run_pair_fn(
            device=device,
            imgA=imgA,
            imgB=imgB,
            prev_data=prev_data,
            image_idA = image_idA,
            image_idB = image_idB,
            **run_pair_kwargs,
        )
        prev_data = prev_result

        flops_total += float(flops)
        runtime_ms_total += float(runtime_ms)

        image_idA += 1
        image_idB += 1

        if i == 0 and compute_depth_gt_from0:
            mk0_01_for_depth_gt = mk0_t.cpu().numpy()

        if mconf is not None and mconf.numel() > 0:
            k = min(int(mconf.numel()), topk)
            idx = torch.topk(mconf, k, 0).indices
            mk0 = mk0_t[idx].cpu().numpy()
            mk1 = mk1_t[idx].cpu().numpy()
            mconf_np = mconf[idx].cpu().numpy()
        else:
            mk0 = mk0_t.cpu().numpy()
            mk1 = mk1_t.cpu().numpy()
            mconf_np = None

        # ---- track linking ----
        if not legacy_linking:
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
                    tr['end_id'] = i + 1
                    tr['confs'].append(conf_val)
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
            curr_points_list = []
            curr_tids_list = []

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

                prev_points = np.array(curr_points_list, dtype=float)
                prev_tids = np.array(curr_tids_list, dtype=int)

                for kk in range(N_cur):
                    if used_current[kk]:
                        continue
                    ptA = mk0[kk]
                    ptB = mk1[kk]
                    conf_val = float(mconf_np[kk]) if mconf_np is not None else 1.0
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

    # ---- 0→N metrics from tracks (epi + depth) ----
    tracks_start_at_0 = sum(1 for tr in tracks.values() if tr['start_id'] == 0)

    endpoints_0, endpoints_N = [], []
    full_track_ids = []
    for tid, tr in tracks.items():
        if tr['start_id'] == 0 and tr['end_id'] == bag_size - 1 and len(tr['points']) == bag_size:
            endpoints_0.append(tr['points'][0])
            endpoints_N.append(tr['points'][-1])
            full_track_ids.append(tid)

            confs = tr.get('confs', [])

            if len(confs) > 1:
                used_confs = np.asarray(confs[1:], dtype=np.float64)
            else:
                used_confs = np.asarray(confs, dtype=np.float64)

            if used_confs.size > 0:
                tr['conf_sum_0_to_N'] = float(used_confs.sum())
                tr['conf_prod_0_to_N'] = float(used_confs.prod())
            else:
                tr['conf_sum_0_to_N'] = 0.0
                tr['conf_prod_0_to_N'] = 0.0


    endpoints_0 = np.array(endpoints_0, dtype=float)
    endpoints_N = np.array(endpoints_N, dtype=float)

    if endpoints_0.shape[0] > 0:
        errs_0N = _compute_symmetric_epi_errors_for_two_cams(
            endpoints_0, endpoints_N, cams[0], cams[-1], device=device
        )

        for idx, tid in enumerate(full_track_ids):
            tracks[tid]['epi_err_0N'] = float(errs_0N[idx])

        n_total_0N = int(errs_0N.size)
        n_correct_0N_1e4 = int((errs_0N < 1e-4).sum())
        n_correct_0N_5e4 = int((errs_0N < 5e-4).sum())
        prec_0N_1e4 = float(n_correct_0N_1e4 / n_total_0N) if n_total_0N else 0.0
        prec_0N_5e4 = float(n_correct_0N_5e4 / n_total_0N) if n_total_0N else 0.0

        R_gt_0N, t_gt_0N = _relative_pose(cams[0], cams[-1])
        R_err_0N = float('nan')
        t_err_0N = float('nan')

        pose0N = {
            'attempted': False,
            'success': False,
            'n_corr': int(endpoints_0.shape[0]),
            'best_n_inliers': None,
            'best_pose_err_deg': None,
            'pose_consistent': False,
        }

        if endpoints_0.shape[0] >= min_corr_0N:

            pose0N['attempted'] = True

            best_pose_err = float('inf')
            best_r_err = float('nan')
            best_t_err = float('nan')
            best_n_inl = None

            for _ in range(1):

                perm = np.random.permutation(len(endpoints_0))
                pts0 = endpoints_0[perm]
                ptsN = endpoints_N[perm]

                est0N = estimate_lo_pose(
                    pts0, ptsN,
                    cams[0].K, cams[-1].K,
                    pixel_thr, conf=conf
                )

                if est0N.get("success", False):

                    M = est0N.get("M_0to_N", est0N.get("M_0to1"))

                    inlier_mask = est0N.get("inliers", None)
                    if inlier_mask is not None:
                        if hasattr(inlier_mask, "cpu"):
                            inlier_mask = inlier_mask.cpu().numpy()
                        n_inl = int(np.sum(inlier_mask))
                    else:
                        n_inl = None

                    t_error, r_error = relative_pose_error(
                        np.block([[R_gt_0N, t_gt_0N.reshape(3, 1)],
                                [np.zeros((1, 3)), np.array([[1.0]])]]),
                        M.R, M.t,
                        ignore_gt_t_thr=0.0
                    )

                    pose_err = max(r_error, t_error)

                    if pose_err < best_pose_err:
                        best_pose_err = pose_err
                        best_r_err = r_error
                        best_t_err = t_error
                        best_n_inl = n_inl

            if best_pose_err < float('inf'):
                R_err_0N = float(best_r_err)
                t_err_0N = float(best_t_err)

                pose0N['success'] = True
                pose0N['best_pose_err_deg'] = best_pose_err
                pose0N['best_n_inliers'] = best_n_inl

                if best_n_inl is not None:
                    pose0N['pose_consistent'] = (
                        best_n_inl >= min_inliers_0N and
                        best_pose_err <= max_pose_err_0N_deg
                    )

        n_tracks_full_0N = int(endpoints_0.shape[0])
        survival_frac = float(n_tracks_full_0N / tracks_start_at_0) if tracks_start_at_0 > 0 else 0.0




        reproj_errs_0N_end_one = []
        reproj_errs_0N_end_sym = []

        if depth_dir is not None and depths[0] is not None and depths[-1] is not None:
            for p0, pN in zip(endpoints_0, endpoints_N):
                e_fwd, valid_fwd, e_bwd, valid_bwd = _compute_depth_errors_between_two_views(
                    p_src=p0,
                    p_tgt=pN,
                    depth_src=depths[0],
                    depth_tgt=depths[-1],
                    cam_src=cams[0],
                    cam_tgt=cams[-1],
                    use_rel_err_gate=False,
                )

                if valid_fwd:
                    reproj_errs_0N_end_one.append(e_fwd)
                if valid_fwd and valid_bwd:
                    reproj_errs_0N_end_sym.append(0.5 * (e_fwd + e_bwd))

        thr_list = [1, 3, 5, 10]

        if len(reproj_errs_0N_end_one) > 0:
            errs = np.asarray(reproj_errs_0N_end_one, dtype=np.float32)
            n_valid_end_one = int(errs.size)
            counts_end_one = [int((errs <= t).sum()) for t in thr_list]
            ratios_end_one = [c / n_valid_end_one for c in counts_end_one]
            median_end_one = float(np.median(errs))
        else:
            n_valid_end_one = 0
            counts_end_one = [0] * len(thr_list)
            ratios_end_one = [0.0] * len(thr_list)
            median_end_one = float('nan')

        if len(reproj_errs_0N_end_sym) > 0:
            errs_sym = np.asarray(reproj_errs_0N_end_sym, dtype=np.float32)
            n_valid_end_sym = int(errs_sym.size)
            counts_end_sym = [int((errs_sym <= t).sum()) for t in thr_list]
            ratios_end_sym = [c / n_valid_end_sym for c in counts_end_sym]
            median_end_sym = float(np.median(errs_sym))
        else:
            n_valid_end_sym = 0
            counts_end_sym = [0] * len(thr_list)
            ratios_end_sym = [0.0] * len(thr_list)
            median_end_sym = float('nan')


        thr_list = [1, 3, 5, 10]

        depth_edge_errs_0N_one = []
        depth_edge_errs_0N_sym = []

        num_steps = bag_size - 1
        track_surv_counts_one = {t: [0] * (num_steps + 1) for t in thr_list}
        track_surv_counts_sym = {t: [0] * (num_steps + 1) for t in thr_list}

        if depth_dir is not None:
            for tid in full_track_ids:
                tr = tracks[tid]
                pts = np.asarray(tr['points'], dtype=np.float32)  # (bag_size, 2)

                alive_one = {t: True for t in thr_list}
                alive_sym = {t: True for t in thr_list}

                for t in thr_list:
                    track_surv_counts_one[t][0] += 1
                    track_surv_counts_sym[t][0] += 1

                for step in range(num_steps):
                    f = step
                    f_next = f + 1

                    p_f = pts[f]
                    p_n = pts[f_next]

                    depth_f = depths[f] if f < len(depths) else None
                    depth_n = depths[f_next] if f_next < len(depths) else None
                    cam_f = cams[f]
                    cam_n = cams[f_next]

                    e_fwd, valid_fwd, e_bwd, valid_bwd = _compute_depth_errors_between_two_views(
                        p_src=p_f,
                        p_tgt=p_n,
                        depth_src=depth_f,
                        depth_tgt=depth_n,
                        cam_src=cam_f,
                        cam_tgt=cam_n,
                        use_rel_err_gate=False
                    )

                    if valid_fwd:
                        depth_edge_errs_0N_one.append(e_fwd)
                    if valid_fwd and valid_bwd:
                        depth_edge_errs_0N_sym.append(0.5 * (e_fwd + e_bwd))

                    for t in thr_list:
                        if alive_one[t]:
                            if (not valid_fwd) or (e_fwd > t):
                                alive_one[t] = False
                            else:
                                track_surv_counts_one[t][f_next] += 1

                    for t in thr_list:
                        if alive_sym[t]:
                            if not (valid_fwd and valid_bwd):
                                alive_sym[t] = False
                            else:
                                e_sym = 0.5 * (e_fwd + e_bwd)
                                if e_sym > t:
                                    alive_sym[t] = False
                                else:
                                    track_surv_counts_sym[t][f_next] += 1

        if len(depth_edge_errs_0N_one) > 0:
            errs = np.asarray(depth_edge_errs_0N_one, dtype=np.float32)
            n_valid = int(errs.size)
            counts = [int((errs <= t).sum()) for t in thr_list]
            ratios = [c / n_valid for c in counts]
            median_px = float(np.median(errs))
        else:
            n_valid = 0
            counts = [0] * len(thr_list)
            ratios = [0.0] * len(thr_list)
            median_px = float('nan')

        reproj_errs_0N = [float(e) for e in depth_edge_errs_0N_one]
        reproj_stats_0N = {
            'thr_px': thr_list,
            'counts': counts,
            'ratios': ratios,
            'n_valid': n_valid,
            'median_err_px': median_px,
            'track_counts_per_k': {
                str(t): track_surv_counts_one[t] for t in thr_list
            },
        }

        if len(depth_edge_errs_0N_sym) > 0:
            errs_sym = np.asarray(depth_edge_errs_0N_sym, dtype=np.float32)
            n_valid_sym = int(errs_sym.size)
            counts_sym = [int((errs_sym <= t).sum()) for t in thr_list]
            ratios_sym = [c / n_valid_sym for c in counts_sym]
            median_px_sym = float(np.median(errs_sym))
        else:
            n_valid_sym = 0
            counts_sym = [0] * len(thr_list)
            ratios_sym = [0.0] * len(thr_list)
            median_px_sym = float('nan')

        reproj_errs_0N_sym = [float(e) for e in depth_edge_errs_0N_sym]
        reproj_stats_0N_sym = {
            'thr_px': thr_list,
            'counts': counts_sym,
            'ratios': ratios_sym,
            'n_valid': n_valid_sym,
            'median_err_px': median_px_sym,
            'track_counts_per_k': {
                str(t): track_surv_counts_sym[t] for t in thr_list
            },
        }

        metrics_0N = {
            'tracks_start_at_0': int(tracks_start_at_0),
            'n_tracks_0_to_N': n_tracks_full_0N,
            'track_survival_fraction_0_to_N': survival_frac,
            'precision@1e-4_0_to_N': prec_0N_1e4,
            'n_correct_0_to_N': n_correct_0N_1e4,
           'precision@5e-4_0_to_N': prec_0N_5e4,
            'n_correct_0_to_N_5e-4': n_correct_0N_5e4,
            'n_total_0_to_N': n_total_0N,
            'median_err_0_to_N': float(np.median(errs_0N)),
            'R_err_deg_0N': R_err_0N,
            't_err_deg_0N': t_err_0N,
            'epi_errs_0N': errs_0N.tolist(),
            'reproj_errs_0N_px': reproj_errs_0N,
            'reproj_stats_0N_px': reproj_stats_0N,
            'reproj_errs_0N_px_sym': reproj_errs_0N_sym,
            'reproj_stats_0N_px_sym': reproj_stats_0N_sym,
            'reproj_errs_0N_px_endpoints': [float(e) for e in reproj_errs_0N_end_one],
            'reproj_stats_0N_px_endpoints': {
                'thr_px': thr_list,
                'counts': counts_end_one,
                'ratios': ratios_end_one,
                'n_valid': n_valid_end_one,
                'median_err_px': median_end_one,
            },
            'reproj_errs_0N_px_endpoints_sym': [float(e) for e in reproj_errs_0N_end_sym],
            'reproj_stats_0N_px_endpoints_sym': {
                'thr_px': thr_list,
                'counts': counts_end_sym,
                'ratios': ratios_end_sym,
                'n_valid': n_valid_end_sym,
                'median_err_px': median_end_sym,
            },
            'pose_0N': pose0N,
        }
    else:
        metrics_0N = {
            'tracks_start_at_0': int(tracks_start_at_0),
            'n_tracks_0_to_N': 0,
            'track_survival_fraction_0_to_N': 0.0,
            'precision@1e-4_0_to_N': 0.0,
            'n_correct_0_to_N': 0,
            'n_total_0_to_N': 0,
            'median_err_0_to_N': float('nan'),
            'R_err_deg_0N': float('nan'),
            't_err_deg_0N': float('nan'),
            'epi_errs_0N': [],
            'reproj_errs_0N_px': [],
            'reproj_stats_0N_px': _empty_reproj_stats(),
            'reproj_errs_0N_px_sym': [],
            'reproj_stats_0N_px_sym': _empty_reproj_stats(),
            'pose_0N': None
        }

    # ---- 0→k (k=1..N-1) metrics from tracks（epi + depth） ----
    metrics_0k_list = []
    for k in range(1, bag_size):
        endpoints_0k_0 = []
        endpoints_0k_k = []
        tids_0k = []
        for tid, tr in tracks.items():
            if (tr.get('start_id', -1) == 0 and
                tr.get('end_id', -1) >= k and
                len(tr.get('points', [])) >= (k + 1)):
                endpoints_0k_0.append(tr['points'][0])
                endpoints_0k_k.append(tr['points'][k])
                tids_0k.append(tid)

        endpoints_0k_0 = np.array(endpoints_0k_0, dtype=float)
        endpoints_0k_k = np.array(endpoints_0k_k, dtype=float)

        if endpoints_0k_0.shape[0] > 0:
            errs_0k = _compute_symmetric_epi_errors_for_two_cams(
                endpoints_0k_0, endpoints_0k_k, cams[0], cams[k], device=device
            )
            n_total_0k = int(errs_0k.size)
            n_correct_0k_1e4 = int((errs_0k < 1e-4).sum())
            n_correct_0k_5e4 = int((errs_0k < 5e-4).sum())
            prec_0k_1e4 = float(n_correct_0k_1e4 / n_total_0k) if n_total_0k else 0.0
            prec_0k_5e4 = float(n_correct_0k_5e4 / n_total_0k) if n_total_0k else 0.0
            median_err_0k = float(np.median(errs_0k))

            thr_list = [1, 3, 5, 10]
            depth_edge_errs_0k_one = []
            depth_edge_errs_0k_sym = []
            track_surv_counts_one = {t: 0 for t in thr_list}
            track_surv_counts_sym = {t: 0 for t in thr_list}

            if depth_dir is not None:
                for tid in tids_0k:
                    tr = tracks[tid]
                    start_id = tr['start_id']
                    pts = np.asarray(tr['points'], dtype=np.float32)

                    if start_id > 0:
                        continue
                    if len(pts) <= k:
                        continue

                    alive_one = {t: True for t in thr_list}
                    alive_sym = {t: True for t in thr_list}

                    for f in range(0, k):  # edges 0→1,1→2,...,(k-1)→k
                        p_f = pts[f]
                        p_n = pts[f + 1]

                        depth_f = depths[f] if f < len(depths) else None
                        depth_n = depths[f + 1] if (f + 1) < len(depths) else None
                        cam_f = cams[f]
                        cam_n = cams[f + 1]

                        e_fwd, valid_fwd, e_bwd, valid_bwd = _compute_depth_errors_between_two_views(
                            p_src=p_f,
                            p_tgt=p_n,
                            depth_src=depth_f,
                            depth_tgt=depth_n,
                            cam_src=cam_f,
                            cam_tgt=cam_n,
                            use_rel_err_gate=False
                        )

                        if f == k - 1:
                            if valid_fwd:
                                depth_edge_errs_0k_one.append(e_fwd)
                            if valid_fwd and valid_bwd:
                                depth_edge_errs_0k_sym.append(0.5 * (e_fwd + e_bwd))

                        for t in thr_list:
                            if alive_one[t]:
                                if (not valid_fwd) or (e_fwd > t):
                                    alive_one[t] = False

                        for t in thr_list:
                            if alive_sym[t]:
                                if not (valid_fwd and valid_bwd):
                                    alive_sym[t] = False
                                else:
                                    e_sym = 0.5 * (e_fwd + e_bwd)
                                    if e_sym > t:
                                        alive_sym[t] = False

                    for t in thr_list:
                        if alive_one[t]:
                            track_surv_counts_one[t] += 1
                        if alive_sym[t]:
                            track_surv_counts_sym[t] += 1

            if len(depth_edge_errs_0k_one) > 0:
                errs_one = np.asarray(depth_edge_errs_0k_one, dtype=np.float32)
                n_valid_one = int(errs_one.size)
                counts_one = [int((errs_one <= t).sum()) for t in thr_list]
                ratios_one = [c / n_valid_one for c in counts_one]
                median_one = float(np.median(errs_one))
            else:
                n_valid_one = 0
                counts_one = [0] * len(thr_list)
                ratios_one = [0.0] * len(thr_list)
                median_one = float('nan')

            depth_stats_0k = {
                'thr_px': thr_list,
                'counts': counts_one,
                'ratios': ratios_one,
                'n_valid': n_valid_one,
                'median_err_px': median_one,
                'track_counts_survival': {
                    str(t): track_surv_counts_one[t] for t in thr_list
                },
            }

            if len(depth_edge_errs_0k_sym) > 0:
                errs_sym = np.asarray(depth_edge_errs_0k_sym, dtype=np.float32)
                n_valid_sym = int(errs_sym.size)
                counts_sym = [int((errs_sym <= t).sum()) for t in thr_list]
                ratios_sym = [c / n_valid_sym for c in counts_sym]
                median_sym = float(np.median(errs_sym))
            else:
                n_valid_sym = 0
                counts_sym = [0] * len(thr_list)
                ratios_sym = [0.0] * len(thr_list)
                median_sym = float('nan')

            depth_stats_0k_sym = {
                'thr_px': thr_list,
                'counts': counts_sym,
                'ratios': ratios_sym,
                'n_valid': n_valid_sym,
                'median_err_px': median_sym,
                'track_counts_survival': {
                    str(t): track_surv_counts_sym[t] for t in thr_list
                },
            }

            metrics_0k_list.append({
                'k': k,
                'n_tracks_0_to_k': int(endpoints_0k_0.shape[0]),
                'precision@1e-4_0_to_k': prec_0k_1e4,
                'n_correct_0_to_k': n_correct_0k_1e4,
                'precision@5e-4_0_to_k': prec_0k_5e4,
                'n_correct_0_to_k_5e-4': n_correct_0k_5e4,
                'n_total_0_to_k': n_total_0k,

                'median_err_0_to_k': median_err_0k,
                'epi_errs_0k': errs_0k.tolist(),
                'reproj_errs_0k_px': [float(e) for e in depth_edge_errs_0k_one],
                'reproj_stats_0k_px': depth_stats_0k,
                'reproj_errs_0k_px_sym': [float(e) for e in depth_edge_errs_0k_sym],
                'reproj_stats_0k_px_sym': depth_stats_0k_sym,
            })
        else:
            metrics_0k_list.append({
                'k': k,
                'n_tracks_0_to_k': 0,
                'precision@1e-4_0_to_k': 0.0,
                'n_correct_0_to_k': 0,
                'n_total_0_to_k': 0,
                'median_err_0_to_k': float('nan'),
                'epi_errs_0k': [],
                'reproj_errs_0k_px': [],
                'reproj_stats_0k_px': _empty_reproj_stats(),
                'reproj_errs_0k_px_sym': [],
                'reproj_stats_0k_px_sym': _empty_reproj_stats(),
            })

    if compute_depth_gt_from0 and mk0_01_for_depth_gt is not None and depths[0] is not None:
        depth_gt_vis = _compute_depth_gt_visibility_from0(
            mk0_01=mk0_01_for_depth_gt,
            depth0=depths[0],
            cams=cams,
            depths=depths,
            mutual_backcheck_thr_px=1.0,
        )
    else:
        depth_gt_vis = None

    return {
        'bag_file': str(bag_file),
        'pairs': pair_results,
        'tracks': {str(tid): tr for tid, tr in tracks.items()},
        'metrics_0_to_N': metrics_0N,
        'metrics_0_to_k_all': metrics_0k_list,
        'flops_total': flops_total,
        'runtime_ms_total': runtime_ms_total,
        'num_pairs': num_pairs,
        'depth_gt_visibility_from0': depth_gt_vis,
    }


# ============================================================
# ============================================================

def aggregate_pair_metrics(all_results, epi_err_thr=1e-4):
    identifiers = []
    R_errs = []
    t_errs = []
    epi_errs_list = []
    for r in all_results:
        for p in r['pairs']:
            identifiers.append(f"{Path(p['imgA']).stem}#{Path(p['imgB']).stem}")
            R_errs.append(p['R_err_deg'])
            t_errs.append(p['t_err_deg'])
            epi_errs_list.append(np.array(p['epi_errs'], dtype=float))

    unq_ids = list(OrderedDict((iden, i) for i, iden in enumerate(identifiers)).values())
    R_errs = np.array(R_errs, dtype=float)[unq_ids]
    t_errs = np.array(t_errs, dtype=float)[unq_ids]

    pose_errors = np.maximum(R_errs, t_errs)
    aucs = {
        f'auc@{thr}': round(float((pose_errors < thr).mean() * 100.0), 2)
        for thr in [5, 10, 20]
    }

    epi_all = np.concatenate([epi_errs_list[i] for i in unq_ids if epi_errs_list[i].size > 0]) \
              if len(unq_ids) > 0 else np.array([], dtype=float)

    if epi_all.size:
        n_total = int(epi_all.size)
        n_correct_1e4 = int((epi_all < 1e-4).sum())
        n_correct_5e4 = int((epi_all < 5e-4).sum())
        prec_1e4 = float(n_correct_1e4 / n_total)
        prec_5e4 = float(n_correct_5e4 / n_total)
    else:
            n_total = 0
            n_correct_1e4 = n_correct_5e4 = 0
            prec_1e4 = prec_5e4 = 0.0

    return {
        **aucs,
        'precision@1e-4': prec_1e4,
        'precision@5e-4': prec_5e4,
        'counts@1e-4': {'correct': n_correct_1e4, 'total': n_total},
        'counts@5e-4': {'correct': n_correct_5e4, 'total': n_total},
        'n_pairs': int(len(unq_ids)),
    }



def aggregate_0N_metrics(all_results, epi_err_thr=1e-4):
    R0N, T0N = [], []
    total_tracks_start0 = 0
    total_tracks_full_0N = 0

    for r in all_results:
        m = r['metrics_0_to_N']
        R_err = m.get('R_err_deg_0N', float('nan'))
        T_err = m.get('t_err_deg_0N', float('nan'))
        if np.isfinite(R_err) and np.isfinite(T_err):
            R0N.append(R_err)
            T0N.append(T_err)

        total_tracks_start0 += m.get('tracks_start_at_0', 0)
        total_tracks_full_0N += m.get('n_tracks_0_to_N', 0)

    auc0N = {}
    if len(R0N) > 0:
        pose_errs_0N = np.maximum(np.array(R0N, dtype=float), np.array(T0N, dtype=float))
        for thr in [5, 10, 20]:
            auc0N[f'auc_0N@{thr}'] = round(float((pose_errs_0N < thr).mean() * 100.0), 2)
    else:
        for thr in [5, 10, 20]:
            auc0N[f'auc_0N@{thr}'] = 0.0

    total_correct_0N = sum(r['metrics_0_to_N']['n_correct_0_to_N'] for r in all_results)
    total_matches_0N = sum(r['metrics_0_to_N']['n_total_0_to_N'] for r in all_results)
    total_correct_0N_5e4 = sum(
        r['metrics_0_to_N'].get('n_correct_0_to_N_5e-4',
                                 r['metrics_0_to_N']['n_correct_0_to_N'])
        for r in all_results
    )
    prec_0N_global = float(total_correct_0N / total_matches_0N) if total_matches_0N else 0.0
    prec_0N_global_5e4 = float(total_correct_0N_5e4 / total_matches_0N) if total_matches_0N else 0.0

    if total_tracks_start0 > 0:
        global_survival_frac = float(total_tracks_full_0N / total_tracks_start0)
    else:
        global_survival_frac = 0.0

    print('auc', auc0N)
    print("Global 0→N Precision@1e-4:", prec_0N_global)
    print('Global correct / total 0→N:', total_correct_0N, '/', total_matches_0N)
    print("Global 0→N Precision@5e-4:", prec_0N_global_5e4)
    print('Global correct / total 0→N (5e-4):', total_correct_0N_5e4, '/', total_matches_0N)

    return {
        **auc0N,
        'global_precision_0N@1e-4_by_counts': prec_0N_global,
        'global_precision_0N@5e-4_by_counts': prec_0N_global_5e4,
        'global_counts_0N': {
            'correct_1e-4': int(total_correct_0N),
            'correct_5e-4': int(total_correct_0N_5e4),
            'total': int(total_matches_0N)
        },
        'global_track_survival_0_to_N': {
            'tracks_full_0_to_N': int(total_tracks_full_0N),
            'tracks_start_at_0': int(total_tracks_start0),
            'fraction_full_0_to_N': global_survival_frac,
        },
        'bags_with_valid_pose_0N': int(len(all_results)),
        'bags_total': int(len(all_results)),
    }


def build_confidence_ranking_curves(
    results: List[dict],
    epi_err_thr: float = 1e-4,
) -> Dict[str, Any]:
    scores_sum = []
    scores_prod = []
    flags = []

    for r in results:
        num_pairs = r['num_pairs']
        tracks_dict = r['tracks']
        for tr in tracks_dict.values():
            if (tr.get('start_id', -1) == 0 and
                tr.get('end_id', -1) == num_pairs and
                len(tr.get('points', [])) == (num_pairs + 1) and
                'epi_err_0N' in tr and
                'conf_sum_0_to_N' in tr and
                'conf_prod_0_to_N' in tr):

                ok = float(tr['epi_err_0N']) < epi_err_thr
                flags.append(1 if ok else 0)
                scores_sum.append(float(tr['conf_sum_0_to_N']))
                scores_prod.append(float(tr['conf_prod_0_to_N']))

    def _build_curve(scores, flags):
        if len(scores) == 0:
            return {
                'num_tracks': 0,
                'cum_correct': [],
                'cum_total': [],
                'cum_precision': [],
            }
        scores = np.asarray(scores, dtype=np.float64)
        flags = np.asarray(flags, dtype=np.int64)
        order = np.argsort(-scores)
        sorted_flags = flags[order]

        cum_correct = np.cumsum(sorted_flags)
        n = len(sorted_flags)
        cum_total = np.arange(1, n + 1, dtype=np.int64)
        cum_precision = (cum_correct / cum_total).astype(float)

        return {
            'num_tracks': int(n),
            'cum_correct': cum_correct.tolist(),
            'cum_total': cum_total.tolist(),
            'cum_precision': cum_precision.tolist(),
        }

    curve_sum = _build_curve(scores_sum, flags)
    curve_prod = _build_curve(scores_prod, flags)

    return {
        'conf_sum_curve_0N': curve_sum,
        'conf_prod_curve_0N': curve_prod,
    }


# ============================================================
# ============================================================

def main():
    args = parse_args()

    # JamMa config
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    config.JAMMA.DET.USE_DET = True
    config.JAMMA.DET.SEARCH_RADIUS = 832 * 2**0.5
    config.JAMMA.DET.FINE_THR = 0.0
    config.JAMMA.USE_COMPILE = False


    profiler = build_profiler(args.profiler_name)
    jamma_model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path,
                           profiler=profiler, dump_dir=args.dump_dir)

    config.JAMMA.DET.USE_DET = False
    jamma_legacy_model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path,
                                  profiler=profiler, dump_dir=args.dump_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')





    # scan bags
    bag_files = sorted(args.subset_dir.glob(f"{args.bag_size}bag_*.txt"))
    if not bag_files:
        raise FileNotFoundError(
            f"No files found in {args.subset_dir} matching pattern {args.bag_size}bag_*.txt"
        )
    all_methods: Dict[str, Callable[..., Any]] = {
        "jamma": run_jamma_pair,
        "det-jamma": run_jamma_pair,
        "jamma_legacy": run_jamma_pair,
    }

    all_methods_kwargs: Dict[str, dict] = {
        "jamma": {"jamma": jamma_model},
        "det-jamma": {"jamma": jamma_model},
        "jamma_legacy": {"jamma": jamma_legacy_model},
    }

    methods: Dict[str, Callable[..., Any]] = {
        name: fn for name, fn in all_methods.items() if name in args.methods
    }
    methods_kwargs: Dict[str, dict] = {
        name: all_methods_kwargs[name] for name in methods.keys()
    }

    if not methods:
        raise ValueError(f"No valid methods selected. Got --methods {args.methods}")

    all_results: Dict[str, list] = {name: [] for name in methods.keys()}
    flops_time: Dict[str, dict] = {
        name: {
            "total_flops_all": 0.0,
            "total_runtime_ms_all": 0.0,
            "total_pairs_all": 0,
        }
        for name in methods.keys()
    }

    # ---- Bag loop: per-bag evaluation ----
    for bf in bag_files:
        print(f"\n[Bag] {bf.name}")
        for method_name, run_pair_fn in methods.items():
            print(f"\n  === Method: {method_name} ===")

            is_legacy = (method_name in ["jamma_legacy"])

            #compute_depth_gt_from0 = (method_name in ["jamma", "eloftr"])
            compute_depth_gt_from0 = (method_name in ["jamma", "det-jamma"])  # DeT-path JamMa variants

            if method_name in ["jamma", "det-jamma", "jamma_legacy"]:
                _ = evaluate_bag_with_tracks(
                        bag_file=bf,
                        bag_size=args.bag_size,
                        dataset_root=args.dataset_root,
                        calib_dir=args.calib_dir,
                        flip_w2c=args.flip_w2c,
                        run_pair_fn=run_pair_fn,
                        run_pair_kwargs=methods_kwargs[method_name],
                        device=device,
                        config=config,
                        epi_threshold=1e-4,
                        topk=20000,
                        legacy_linking=is_legacy,
                        depth_dir=args.depth_dir,
                        compute_depth_gt_from0=compute_depth_gt_from0,
                    )

            res = evaluate_bag_with_tracks(
                bag_file=bf,
                bag_size=args.bag_size,
                dataset_root=args.dataset_root,
                calib_dir=args.calib_dir,
                flip_w2c=args.flip_w2c,
                run_pair_fn=run_pair_fn,
                run_pair_kwargs=methods_kwargs[method_name],
                device=device,
                config=config,
                epi_threshold=1e-4,
                topk=20000,
                legacy_linking=is_legacy,
                depth_dir=args.depth_dir,
                compute_depth_gt_from0=compute_depth_gt_from0,
            )

            # per-pair print
            for p in res['pairs']:
                print(
                    f"  [{p['pair_index']:02d}] {Path(p['imgA']).name} ↔ {Path(p['imgB']).name} | "
                    f"prec={p['precision']:.4f} ({p['n_correct']}/{p['n_total']}) | "
                    f"R_err={p['R_err_deg']:.2f}° | T_err={p['t_err_deg']:.2f}° | "
                    f"median_err={p['median_err']:.2e} | "
                    f"flops={p['flops'] / 1e9:.2f} GMac | time={p['runtime_ms']:.2f} ms"
                )

            m0N = res['metrics_0_to_N']
            print(
                f"  [0→{args.bag_size-1}] "
                f"tracks_full={m0N['n_tracks_0_to_N']}/{m0N['tracks_start_at_0']} "
                f"(survival={m0N['track_survival_fraction_0_to_N']:.3f}) | "
                f"prec@1e-4={m0N['precision@1e-4_0_to_N']:.4f} "
                f"({m0N['n_correct_0_to_N']}/{m0N['n_total_0_to_N']}) | "
                f"prec@5e-4={m0N['precision@5e-4_0_to_N']:.4f} "
                f"({m0N['n_correct_0_to_N_5e-4']}/{m0N['n_total_0_to_N']}) | "
                f"median_err={m0N['median_err_0_to_N']:.2e} | "
                f"R_err={m0N['R_err_deg_0N']:.2f}° | T_err={m0N['t_err_deg_0N']:.2f}° | "
                f"pose0N={m0N['pose_0N']}"
            )
            stats_0N_px = m0N['reproj_stats_0N_px']
            print(
                f"    depth-0→N one-side (k→k+1 edges): n_valid={stats_0N_px['n_valid']} | "
                f"median_px={stats_0N_px['median_err_px']:.2f} | "
                f"<=1px={stats_0N_px['ratios'][0]:.3f}, "
                f"<=3px={stats_0N_px['ratios'][1]:.3f}, "
                f"<=5px={stats_0N_px['ratios'][2]:.3f}, "
                f"<=10px={stats_0N_px['ratios'][3]:.3f}"
            )
            stats_0N_px_sym = m0N.get('reproj_stats_0N_px_sym', _empty_reproj_stats())
            print(
                f"    depth-0↔N symmetric (k↔k+1 mean): n_valid={stats_0N_px_sym['n_valid']} | "
                f"median_px={stats_0N_px_sym['median_err_px']:.2f} | "
                f"<=1px={stats_0N_px_sym['ratios'][0]:.3f}, "
                f"<=3px={stats_0N_px_sym['ratios'][1]:.3f}, "
                f"<=5px={stats_0N_px_sym['ratios'][2]:.3f}, "
                f"<=10px={stats_0N_px_sym['ratios'][3]:.3f}"
            )

            print("  --- 0→k metrics (image0 vs image_k) ---")
            for mk in res['metrics_0_to_k_all']:
                k = mk['k']
                stats_k_px = mk['reproj_stats_0k_px']
                stats_k_px_sym = mk.get('reproj_stats_0k_px_sym', _empty_reproj_stats())
                print(
                    f"    0→{k}: "
                    f"tracks={mk['n_tracks_0_to_k']} | "
                    f"prec@1e-4={mk['precision@1e-4_0_to_k']:.4f} "
                    f"({mk['n_correct_0_to_k']}/{mk['n_total_0_to_k']}) | "
                    f"prec@5e-4={mk['precision@5e-4_0_to_k']:.4f} "
                    f"({mk['n_correct_0_to_k_5e-4']}/{mk['n_total_0_to_k']}) | "
                    f"median_err={mk['median_err_0_to_k']:.2e} | "
                    f"depth n_valid={stats_k_px['n_valid']} "
                    f"<=3px={stats_k_px['ratios'][1]:.3f} | "
                    f"depth-sym n_valid={stats_k_px_sym['n_valid']} "
                    f"sym<=3px={stats_k_px_sym['ratios'][1]:.3f}"
                )

            # Print depth GT visibility for the DeT-path JamMa variants.
            if method_name in ["jamma", "det-jamma"] and res['depth_gt_visibility_from0'] is not None:
                vis = res['depth_gt_visibility_from0']
                print("  --- depth-GT visibility from image0 (JamMa 0→1 mk0 all) ---")
                print(f"    initial_valid_points={vis['num_points_initial']}")
                if vis['num_points_initial'] > 0:
                    print("    [symmetric depth (k↔k+1), base thr≈1px] counts_per_k:")
                    for k_idx, c in enumerate(vis['counts_per_k']):
                        ratio = c / vis['num_points_initial']
                        print(f"      k={k_idx}: count={c}, ratio={ratio:.3f}")

                    if 'counts_per_k_vis_only' in vis:
                        print("    [proj-only 0→k] counts_per_k_vis_only:")
                        for k_idx, c in enumerate(vis['counts_per_k_vis_only']):
                            ratio = c / vis['num_points_initial']
                            print(f"      k={k_idx}: count={c}, ratio={ratio:.3f}")

                    multi = vis.get('multi_thr', None)
                    if multi is not None:
                        thr_px_list = multi['thr_px']
                        counts_multi = multi['counts_per_k']
                        print("    [multi-thr symmetric depth (k↔k+1) survival]")
                        for t_idx, thr in enumerate(thr_px_list):
                            print(f"      thr={thr:.1f}px:")
                            for k_idx, c in enumerate(counts_multi[t_idx]):
                                ratio = c / vis['num_points_initial']
                                print(f"        k={k_idx}: count={c}, ratio={ratio:.3f}")
                else:
                    print("    (no valid initial points)")

            all_results[method_name].append(res)

            flops_time[method_name]["total_flops_all"] += res['flops_total']
            flops_time[method_name]["total_runtime_ms_all"] += res['runtime_ms_total']
            flops_time[method_name]["total_pairs_all"] += res['num_pairs']

    num_bags = len(bag_files)
    equalized_target_n_per_bag: List[int] = []
    for bag_idx in range(num_bags):
        counts = []
        for mname in methods.keys():
            if bag_idx < len(all_results[mname]):
                counts.append(all_results[mname][bag_idx]['metrics_0_to_N']['n_tracks_0_to_N'])
        if counts:
            equalized_target_n_per_bag.append(int(min(counts)))
        else:
            equalized_target_n_per_bag.append(0)

    summary: Dict[str, dict] = {}
    epi_thr_0N = 1e-4
    summary['dataset_name'] = args.dataset_name
    summary['scene_name'] = args.scene_name

    for method_name, results in all_results.items():
        print(f"\n=== Global Summary ({method_name}) ===")

        pair_summary = aggregate_pair_metrics(results, epi_err_thr=1e-4)
        total_matches = sum(p['n_total'] for r in results for p in r['pairs'])
        total_correct_1e4 = sum(p['n_correct'] for r in results for p in r['pairs'])
        total_correct_5e4 = 0
        for r in results:
            for p in r['pairs']:
                if p.get('epi_errs', None) is not None:
                    errs = np.array(p['epi_errs'], dtype=float)
                    total_correct_5e4 += int((errs < 5e-4).sum())

        pair_summary.update({
            'global_precision@1e-4_by_counts':
                float(total_correct_1e4 / total_matches) if total_matches else 0.0,
            'global_precision@5e-4_by_counts':
                float(total_correct_5e4 / total_matches) if total_matches else 0.0,
            'global_counts': {
                'correct_1e-4': int(total_correct_1e4),
                'correct_5e-4': int(total_correct_5e4),
                'total': int(total_matches),
            },
        })

        ft = flops_time[method_name]
        if ft["total_pairs_all"] > 0:
            avg_flops_GMac = (ft["total_flops_all"] / ft["total_pairs_all"]) / 1e9
            avg_time_ms = ft["total_runtime_ms_all"] / ft["total_pairs_all"]
        else:
            avg_flops_GMac = 0.0
            avg_time_ms = 0.0

        flops_time_summary = {
            'total_pairs': int(ft["total_pairs_all"]),
            'total_flops_GMac': float(ft["total_flops_all"] / 1e9),
            'total_runtime_ms': float(ft["total_runtime_ms_all"]),
            'avg_flops_per_pair_GMac': float(avg_flops_GMac),
            'avg_runtime_per_pair_ms': float(avg_time_ms),
        }

        summary_0N = aggregate_0N_metrics(results, epi_err_thr=1e-4)

        global_0k_summary = {}
        if results:
            k_values = sorted({ mk['k'] for r in results for mk in r['metrics_0_to_k_all'] })
            for k in k_values:
                total_correct_k_1e4 = 0
                total_correct_k_5e4 = 0
                total_total_k = 0
                all_errs_k = []

                for r in results:
                    mk_list = r['metrics_0_to_k_all']
                    mk_for_k = next((x for x in mk_list if x['k'] == k), None)
                    if mk_for_k is None:
                        continue
                    total_correct_k_1e4 += mk_for_k['n_correct_0_to_k']
                    if mk_for_k['epi_errs_0k']:
                        errs_arr = np.array(mk_for_k['epi_errs_0k'], dtype=float)
                        total_correct_k_5e4 += int((errs_arr < 5e-4).sum())
                    total_total_k += mk_for_k['n_total_0_to_k']
                    if mk_for_k['epi_errs_0k']:
                        all_errs_k.append(np.array(mk_for_k['epi_errs_0k'], dtype=float))

                if total_total_k > 0:
                    prec_k_1e4 = float(total_correct_k_1e4 / total_total_k)
                    prec_k_5e4 = float(total_correct_k_5e4 / total_total_k)
                else:
                    prec_k_1e4 = 0.0
                    prec_k_5e4 = 0.0

                if all_errs_k:
                    errs_concat = np.concatenate(all_errs_k)
                    median_err_k = float(np.median(errs_concat))
                else:
                    median_err_k = float('nan')

                global_0k_summary[str(k)] = {
                    'precision@1e-4_0_to_k': prec_k_1e4,
                    'precision@5e-4_0_to_k': prec_k_5e4,
                    'global_counts_0_to_k': {
                        'correct_1e-4': int(total_correct_k_1e4),
                        'correct_5e-4': int(total_correct_k_5e4),
                        'total': int(total_total_k),
                    },
                    'median_err_0_to_k': median_err_k,
                }

        if global_0k_summary:
            print("\nGlobal 0→k metrics (epi, image0 vs image_k):")
            for k_str in sorted(global_0k_summary.keys(), key=lambda x: int(x)):
                gk = global_0k_summary[k_str]
                print(
                    f"  0→{k_str}: "
                    f"prec@1e-4={gk['precision@1e-4_0_to_k']:.4f} "
                    f"({gk['global_counts_0_to_k']['correct_1e-4']}/"
                    f"{gk['global_counts_0_to_k']['total']}) | "
                    f"prec@5e-4={gk['precision@5e-4_0_to_k']:.4f} "
                    f"({gk['global_counts_0_to_k']['correct_5e-4']}/"
                    f"{gk['global_counts_0_to_k']['total']}) | "
                    f"median_err={gk['median_err_0_to_k']:.2e}"
                )

        thr_px = [1, 3, 5, 10]
        depth_counts_0N = [0] * len(thr_px)
        depth_n_valid_0N = 0
        all_reproj_errs_0N = []

        for r in results:
            m = r['metrics_0_to_N']
            stats = m.get('reproj_stats_0N_px', None)
            if stats is None:
                continue
            depth_n_valid_0N += stats.get('n_valid', 0)
            cs = stats.get('counts', [0] * len(thr_px))
            for i_t in range(len(thr_px)):
                depth_counts_0N[i_t] += int(cs[i_t])
            if m.get('reproj_errs_0N_px', None):
                all_reproj_errs_0N.append(np.array(m['reproj_errs_0N_px'], dtype=float))

        if depth_n_valid_0N > 0:
            depth_ratios_0N = [c / depth_n_valid_0N for c in depth_counts_0N]
            if all_reproj_errs_0N:
                depth_median_0N = float(np.median(np.concatenate(all_reproj_errs_0N)))
            else:
                depth_median_0N = float('nan')
        else:
            depth_ratios_0N = [0.0] * len(thr_px)
            depth_median_0N = float('nan')

        depth_0N_summary = {
            'thr_px': thr_px,
            'counts': depth_counts_0N,
            'ratios': depth_ratios_0N,
            'n_valid': depth_n_valid_0N,
            'median_err_px': depth_median_0N,
        }

        c0, c1, c2, c3 = depth_0N_summary['counts']
        r0, r1, r2, r3 = depth_0N_summary['ratios']

        depth_counts_0N_sym = [0] * len(thr_px)
        depth_n_valid_0N_sym = 0
        all_reproj_errs_0N_sym = []

        for r in results:
            m = r['metrics_0_to_N']
            stats_sym = m.get('reproj_stats_0N_px_sym', None)
            if stats_sym is None:
                continue
            depth_n_valid_0N_sym += stats_sym.get('n_valid', 0)
            cs_sym = stats_sym.get('counts', [0] * len(thr_px))
            for i_t in range(len(thr_px)):
                depth_counts_0N_sym[i_t] += int(cs_sym[i_t])
            if m.get('reproj_errs_0N_px_sym', None):
                all_reproj_errs_0N_sym.append(np.array(m['reproj_errs_0N_px_sym'], dtype=float))

        if depth_n_valid_0N_sym > 0:
            depth_ratios_0N_sym = [c / depth_n_valid_0N_sym for c in depth_counts_0N_sym]
            if all_reproj_errs_0N_sym:
                depth_median_0N_sym = float(np.median(np.concatenate(all_reproj_errs_0N_sym)))
            else:
                depth_median_0N_sym = float('nan')
        else:
            depth_ratios_0N_sym = [0.0] * len(thr_px)
            depth_median_0N_sym = float('nan')

        depth_0N_summary_sym = {
            'thr_px': thr_px,
            'counts': depth_counts_0N_sym,
            'ratios': depth_ratios_0N_sym,
            'n_valid': depth_n_valid_0N_sym,
            'median_err_px': depth_median_0N_sym,
        }

        c0s, c1s, c2s, c3s = depth_0N_summary_sym['counts']
        r0s, r1s, r2s, r3s = depth_0N_summary_sym['ratios']


        thr_px = [1, 3, 5, 10]

        depth_counts_0N_tracks_one = [0] * len(thr_px)
        depth_tracks_total_0N = 0

        depth_counts_0N_tracks_sym = [0] * len(thr_px)

        all_reproj_errs_0N_end_one = []
        all_reproj_errs_0N_end_sym = []

        for r in results:
            m = r['metrics_0_to_N']

            n_tracks_full = m.get('n_tracks_0_to_N', 0)
            if n_tracks_full == 0:
                continue

            stats_end = m.get('reproj_stats_0N_px_endpoints', None)
            stats_end_sym = m.get('reproj_stats_0N_px_endpoints_sym', None)

            if stats_end is None or stats_end_sym is None:
                depth_tracks_total_0N += n_tracks_full
                continue

            depth_tracks_total_0N += n_tracks_full

            cs_one = stats_end.get('counts', [0] * len(thr_px))
            cs_sym = stats_end_sym.get('counts', [0] * len(thr_px))

            for i_t in range(len(thr_px)):
                depth_counts_0N_tracks_one[i_t] += int(cs_one[i_t])
                depth_counts_0N_tracks_sym[i_t] += int(cs_sym[i_t])

            if m.get('reproj_errs_0N_px_endpoints', None):
                all_reproj_errs_0N_end_one.append(
                    np.array(m['reproj_errs_0N_px_endpoints'], dtype=float)
                )
            if m.get('reproj_errs_0N_px_endpoints_sym', None):
                all_reproj_errs_0N_end_sym.append(
                    np.array(m['reproj_errs_0N_px_endpoints_sym'], dtype=float)
                )

        if depth_tracks_total_0N > 0:
            depth_ratios_0N_tracks_one = [
                c / depth_tracks_total_0N for c in depth_counts_0N_tracks_one
            ]
            depth_ratios_0N_tracks_sym = [
                c / depth_tracks_total_0N for c in depth_counts_0N_tracks_sym
            ]
        else:
            depth_ratios_0N_tracks_one = [0.0] * len(thr_px)
            depth_ratios_0N_tracks_sym = [0.0] * len(thr_px)

        if all_reproj_errs_0N_end_one:
            depth_median_0N_end_one = float(
                np.median(np.concatenate(all_reproj_errs_0N_end_one))
            )
        else:
            depth_median_0N_end_one = float('nan')

        if all_reproj_errs_0N_end_sym:
            depth_median_0N_end_sym = float(
                np.median(np.concatenate(all_reproj_errs_0N_end_sym))
            )
        else:
            depth_median_0N_end_sym = float('nan')

        depth_0N_tracks_summary_one = {
            'thr_px': thr_px,
            'counts': depth_counts_0N_tracks_one,
            'ratios': depth_ratios_0N_tracks_one,         # counts / depth_tracks_total_0N
            'n_tracks_total_0N': int(depth_tracks_total_0N),
            'median_err_px_valid_only': depth_median_0N_end_one,
        }

        depth_0N_tracks_summary_sym = {
            'thr_px': thr_px,
            'counts': depth_counts_0N_tracks_sym,
            'ratios': depth_ratios_0N_tracks_sym,
            'n_tracks_total_0N': int(depth_tracks_total_0N),
            'median_err_px_valid_only': depth_median_0N_end_sym,
        }


        depth_0k_summary = {}
        if results:
            k_values = sorted({ mk['k'] for r in results for mk in r['metrics_0_to_k_all'] })
            for k in k_values:
                counts_k = [0] * len(thr_px)
                n_valid_k = 0
                all_errs_k_px = []
                global_track_surv_k = {str(t): 0 for t in thr_px}

                for r in results:
                    mk_list = r['metrics_0_to_k_all']
                    mk_for_k = next((x for x in mk_list if x['k'] == k), None)
                    if mk_for_k is None:
                        continue
                    stats_k = mk_for_k.get('reproj_stats_0k_px', None)
                    if stats_k is None:
                        continue
                    n_valid_k += stats_k.get('n_valid', 0)
                    cs_k = stats_k.get('counts', [0] * len(thr_px))
                    for i_t in range(len(thr_px)):
                        counts_k[i_t] += int(cs_k[i_t])

                    if mk_for_k.get('reproj_errs_0k_px', None):
                        all_errs_k_px.append(np.array(mk_for_k['reproj_errs_0k_px'], dtype=float))

                    tcs = stats_k.get('track_counts_survival', None)
                    if tcs is not None:
                        for t_key, c in tcs.items():
                            t_str = str(t_key)
                            global_track_surv_k[t_str] = global_track_surv_k.get(t_str, 0) + int(c)

                if n_valid_k > 0:
                    ratios_k = [c / n_valid_k for c in counts_k]
                    if all_errs_k_px:
                        median_k_px = float(np.median(np.concatenate(all_errs_k_px)))
                    else:
                        median_k_px = float('nan')
                else:
                    ratios_k = [0.0] * len(thr_px)
                    median_k_px = float('nan')

                depth_0k_summary[str(k)] = {
                    'thr_px': thr_px,
                    'counts': counts_k,
                    'ratios': ratios_k,
                    'n_valid': n_valid_k,
                    'median_err_px': median_k_px,
                    'track_counts_survival_global': global_track_surv_k,
                }

        if depth_0k_summary:
            print("\nGlobal depth-based 0→k reprojection metrics (one-side edges):")
            for k_str in sorted(depth_0k_summary.keys(), key=lambda x: int(x)):
                dk = depth_0k_summary[k_str]
                c0k, c1k, c2k, c3k = dk['counts']
                r0k, r1k, r2k, r3k = dk['ratios']

        depth_0k_summary_sym = {}
        if results:
            k_values = sorted({ mk['k'] for r in results for mk in r['metrics_0_to_k_all'] })
            for k in k_values:
                counts_k_sym = [0] * len(thr_px)
                n_valid_k_sym = 0
                all_errs_k_px_sym = []
                global_track_surv_k_sym = {str(t): 0 for t in thr_px}

                for r in results:
                    mk_list = r['metrics_0_to_k_all']
                    mk_for_k = next((x for x in mk_list if x['k'] == k), None)
                    if mk_for_k is None:
                        continue
                    stats_k_sym = mk_for_k.get('reproj_stats_0k_px_sym', None)
                    if stats_k_sym is None:
                        continue
                    n_valid_k_sym += stats_k_sym.get('n_valid', 0)
                    cs_k_sym = stats_k_sym.get('counts', [0] * len(thr_px))
                    for i_t in range(len(thr_px)):
                        counts_k_sym[i_t] += int(cs_k_sym[i_t])

                    if mk_for_k.get('reproj_errs_0k_px_sym', None):
                        all_errs_k_px_sym.append(
                            np.array(mk_for_k['reproj_errs_0k_px_sym'], dtype=float)
                        )

                    tcs_sym = stats_k_sym.get('track_counts_survival', None)
                    if tcs_sym is not None:
                        for t_key, c in tcs_sym.items():
                            t_str = str(t_key)
                            global_track_surv_k_sym[t_str] = (
                                global_track_surv_k_sym.get(t_str, 0) + int(c)
                            )

                if n_valid_k_sym > 0:
                    ratios_k_sym = [c / n_valid_k_sym for c in counts_k_sym]
                    if all_errs_k_px_sym:
                        median_k_px_sym = float(np.median(np.concatenate(all_errs_k_px_sym)))
                    else:
                        median_k_px_sym = float('nan')
                else:
                    ratios_k_sym = [0.0] * len(thr_px)
                    median_k_px_sym = float('nan')

                depth_0k_summary_sym[str(k)] = {
                    'thr_px': thr_px,
                    'counts': counts_k_sym,
                    'ratios': ratios_k_sym,
                    'n_valid': n_valid_k_sym,
                    'median_err_px': median_k_px_sym,
                    'track_counts_survival_global': global_track_surv_k_sym,
                }

        if depth_0k_summary_sym:
            print("\nGlobal depth-based 0↔k symmetric reprojection metrics (edge mean):")
            for k_str in sorted(depth_0k_summary_sym.keys(), key=lambda x: int(x)):
                dk = depth_0k_summary_sym[k_str]
                c0k, c1k, c2k, c3k = dk['counts']
                r0k, r1k, r2k, r3k = dk['ratios']

        thr_px = [1, 3, 5, 10]
        depth_counts_0N_end = [0] * len(thr_px)
        depth_n_valid_0N_end = 0
        all_reproj_errs_0N_end = []

        for r in results:
            m = r['metrics_0_to_N']
            stats_end = m.get('reproj_stats_0N_px_endpoints', None)
            if stats_end is None:
                continue
            depth_n_valid_0N_end += stats_end.get('n_valid', 0)
            cs = stats_end.get('counts', [0] * len(thr_px))
            for i_t in range(len(thr_px)):
                depth_counts_0N_end[i_t] += int(cs[i_t])
            if m.get('reproj_errs_0N_px_endpoints', None):
                all_reproj_errs_0N_end.append(
                    np.array(m['reproj_errs_0N_px_endpoints'], dtype=float)
                )

        if depth_n_valid_0N_end > 0:
            depth_ratios_0N_end = [c / depth_n_valid_0N_end for c in depth_counts_0N_end]
            if all_reproj_errs_0N_end:
                depth_median_0N_end = float(np.median(np.concatenate(all_reproj_errs_0N_end)))
            else:
                depth_median_0N_end = float('nan')
        else:
            depth_ratios_0N_end = [0.0] * len(thr_px)
            depth_median_0N_end = float('nan')

        depth_0N_endpoints_summary = {
            'thr_px': thr_px,
            'counts': depth_counts_0N_end,
            'ratios': depth_ratios_0N_end,
            'n_valid': depth_n_valid_0N_end,
            'median_err_px': depth_median_0N_end,
        }


        # ---------- equalized 0→N precision (epi) ----------
        equalized = {}
        total_eq_correct_sum = 0
        total_eq_den_sum = 0
        total_eq_correct_prod = 0
        total_eq_den_prod = 0

        for bag_idx, r in enumerate(results):
            if bag_idx >= len(equalized_target_n_per_bag):
                continue
            target_n = equalized_target_n_per_bag[bag_idx]
            if target_n <= 0:
                continue

            tracks_dict = r['tracks']
            full_tracks = []
            for tid_str, tr in tracks_dict.items():
                if (tr.get('start_id', -1) == 0 and
                    tr.get('end_id', -1) == r['num_pairs'] and
                    len(tr.get('points', [])) == (r['num_pairs'] + 1) and
                    'epi_err_0N' in tr and
                    'conf_sum_0_to_N' in tr and
                    'conf_prod_0_to_N' in tr):
                    full_tracks.append(tr)

            if len(full_tracks) == 0:
                total_eq_den_sum += target_n
                total_eq_den_prod += target_n
                continue

            num_chosen = min(target_n, len(full_tracks))

            full_tracks_sorted_sum = sorted(
                full_tracks,
                key=lambda tr: tr['conf_sum_0_to_N'],
                reverse=True
            )
            chosen_sum = full_tracks_sorted_sum[:num_chosen]
            correct_sum = sum(1 for tr in chosen_sum if tr['epi_err_0N'] < epi_thr_0N)
            total_eq_correct_sum += correct_sum
            total_eq_den_sum += target_n

            full_tracks_sorted_prod = sorted(
                full_tracks,
                key=lambda tr: tr['conf_prod_0_to_N'],
                reverse=True
            )
            chosen_prod = full_tracks_sorted_prod[:num_chosen]
            correct_prod = sum(1 for tr in chosen_prod if tr['epi_err_0N'] < epi_thr_0N)
            total_eq_correct_prod += correct_prod
            total_eq_den_prod += target_n

        if total_eq_den_sum > 0:
            equalized['global_precision_0N_equalized_min_tracks_conf_sum'] = \
                float(total_eq_correct_sum / total_eq_den_sum)
            equalized['global_counts_0N_equalized_min_tracks_conf_sum'] = {
                'correct': int(total_eq_correct_sum),
                'total': int(total_eq_den_sum),
            }
        else:
            equalized['global_precision_0N_equalized_min_tracks_conf_sum'] = 0.0
            equalized['global_counts_0N_equalized_min_tracks_conf_sum'] = {
                'correct': 0,
                'total': 0,
            }

        if total_eq_den_prod > 0:
            equalized['global_precision_0N_equalized_min_tracks_conf_prod'] = \
                float(total_eq_correct_prod / total_eq_den_prod)
            equalized['global_counts_0N_equalized_min_tracks_conf_prod'] = {
                'correct': int(total_eq_correct_prod),
                'total': int(total_eq_den_prod),
            }
        else:
            equalized['global_precision_0N_equalized_min_tracks_conf_prod'] = 0.0
            equalized['global_counts_0N_equalized_min_tracks_conf_prod'] = {
                'correct': 0,
                'total': 0,
            }

        # ---------- Global depth GT visibility from image0 (JamMa 0→1 mk0 all) ----------
        depth_gt_visibility_global = None
        if method_name in ["jamma", "det-jamma"]:
            total_initial_points = 0
            total_counts_per_k = [0] * args.bag_size           # symmetric depth base thr≈1px
            total_counts_per_k_vis_only = [0] * args.bag_size  # proj-only 0→k

            global_thr_px = None
            global_counts_per_k_multi = None  # shape (T, bag_size)

            for r in results:
                vis = r.get('depth_gt_visibility_from0', None)
                if vis is None:
                    continue
                total_initial_points += vis['num_points_initial']

                cp = vis['counts_per_k']
                for k_idx in range(min(len(cp), args.bag_size)):
                    total_counts_per_k[k_idx] += cp[k_idx]

                cp_vo = vis.get('counts_per_k_vis_only', None)
                if cp_vo is not None:
                    for k_idx in range(min(len(cp_vo), args.bag_size)):
                        total_counts_per_k_vis_only[k_idx] += cp_vo[k_idx]

                multi = vis.get('multi_thr', None)
                if multi is not None:
                    thr_px_list = multi['thr_px']
                    counts_multi = multi['counts_per_k']  # list[T][num_cams]
                    if global_thr_px is None:
                        global_thr_px = list(thr_px_list)
                        global_counts_per_k_multi = [
                            [0] * args.bag_size for _ in range(len(global_thr_px))
                        ]
                    else:
                        if list(thr_px_list) != global_thr_px:
                            print("[WARN] thr_px list differs across bags; using first one.")
                    for t_idx, _ in enumerate(global_thr_px):
                        counts_k_bag = counts_multi[t_idx]
                        for k_idx in range(min(len(counts_k_bag), args.bag_size)):
                            global_counts_per_k_multi[t_idx][k_idx] += counts_k_bag[k_idx]

            depth_gt_visibility_global = {
                'total_initial_points': int(total_initial_points),
                'total_counts_per_k': [int(c) for c in total_counts_per_k],
                'total_counts_per_k_vis_only': [int(c) for c in total_counts_per_k_vis_only],
            }
            if global_thr_px is not None and global_counts_per_k_multi is not None:
                depth_gt_visibility_global['multi_thr'] = {
                    'thr_px': [float(t) for t in global_thr_px],
                    'total_counts_per_k': [
                        [int(c) for c in row] for row in global_counts_per_k_multi
                    ],
                }

            if total_initial_points > 0:
                for k_idx, c in enumerate(depth_gt_visibility_global['total_counts_per_k']):
                    ratio = c / total_initial_points if total_initial_points > 0 else 0.0

                for k_idx, c in enumerate(depth_gt_visibility_global['total_counts_per_k_vis_only']):
                    ratio = c / total_initial_points if total_initial_points > 0 else 0.0

                multi_g = depth_gt_visibility_global.get('multi_thr', None)
                if multi_g is not None:
                    thr_px_list = multi_g['thr_px']
                    counts_multi = multi_g['total_counts_per_k']
                    for t_idx, thr in enumerate(thr_px_list):
                        for k_idx, c in enumerate(counts_multi[t_idx]):
                            ratio = c / total_initial_points if total_initial_points > 0 else 0.0
            else:
                print("  (no valid initial points for depth GT visibility)")

        conf_rank_stats_1e4 = build_confidence_ranking_curves(
            results=results,
            epi_err_thr=1e-4,
        )
        conf_rank_stats_5e4 = build_confidence_ranking_curves(
            results=results,
            epi_err_thr=5e-4,
        )

        summary[method_name] = {
            'pairs_summary': pair_summary,
            '0N_summary': summary_0N,
            '0k_summary': global_0k_summary,
            'flops_time_summary': flops_time_summary,
            'depth_0N_summary': depth_0N_summary,
            'depth_0k_summary': depth_0k_summary,
            # symmetric depth summary
            'depth_0N_summary_sym': depth_0N_summary_sym,
            'depth_0k_summary_sym': depth_0k_summary_sym,
            '0N_confidence_ranking': {
                '1e-4': conf_rank_stats_1e4,
                '5e-4': conf_rank_stats_5e4,
            },
            'depth_0N_endpoints_summary': depth_0N_endpoints_summary,

            'depth_0N_tracks_summary_one_side': depth_0N_tracks_summary_one,
            'depth_0N_tracks_summary_sym': depth_0N_tracks_summary_sym,
        }

        if equalized:
            summary[method_name]['0N_equalized_min_tracks'] = equalized
        if depth_gt_visibility_global is not None:
            summary[method_name]['depth_gt_visibility_from0_global'] = depth_gt_visibility_global
        topK = 19085

        # ---- conf_sum (1e-4) ----
        cr_sum = conf_rank_stats_1e4['conf_sum_curve_0N']
        if cr_sum['num_tracks'] > 0:
            K = min(topK, cr_sum['num_tracks'])
            correct_K = cr_sum['cum_correct'][K - 1]
            total_K   = cr_sum['cum_total'][K - 1]   # == K
            prec_K    = correct_K / total_K if total_K > 0 else 0.0

            print(f"\n  [conf_sum] 0→N top-{K}: "
                f"correct={correct_K} / total={total_K}  "
                f"precision={prec_K:.4f}")

        # ---- conf_prod (1e-4) ----
        cr_prod = conf_rank_stats_1e4['conf_prod_curve_0N']
        if cr_prod['num_tracks'] > 0:
            K = min(topK, cr_prod['num_tracks'])
            correct_K = cr_prod['cum_correct'][K - 1]
            total_K   = cr_prod['cum_total'][K - 1]
            prec_K    = correct_K / total_K if total_K > 0 else 0.0

            print(f"  [conf_prod] 0→N top-{K}: "
                f"correct={correct_K} / total={total_K}  "
                f"precision={prec_K:.4f}")


        json.dumps(summary[method_name], indent=2)

        if ft["total_pairs_all"] > 0:
            print(f"\nAveraged matching time over {ft['total_pairs_all']} pairs ({method_name}): "
                  f"{avg_time_ms:.2f} ms")
            print(f"Averaged FLOPs per pair ({method_name}): {avg_flops_GMac:.2f} GMac")

    if args.save_json:
        out_json = {
            'per_bag': all_results,
            'summary': summary,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(out_json, indent=2, default=_json_default))
        print(f"Full results saved to {args.save_json}")

    if args.save_summary_json:
        args.save_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_summary_json.write_text(json.dumps(summary, indent=2, default=_json_default))
        print(f"Global summary saved to {args.save_summary_json}")


if __name__ == "__main__":
    main()
