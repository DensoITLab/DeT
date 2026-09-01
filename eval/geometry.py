from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch

from eval.records import CameraParams
from src.utils.metrics import symmetric_epipolar_distance


def _read_cam_from_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        if "K" in f and "R" in f and ("T" in f or "t" in f):
            K = np.asarray(f["K"])
            R = np.asarray(f["R"])
            t = np.asarray(f["T"] if "T" in f else f["t"])
            return K.reshape(3, 3), R.reshape(3, 3), t.reshape(3)

        for key in f.keys():
            group = f[key]
            if isinstance(group, h5py.Group) and "K" in group and "R" in group and ("T" in group or "t" in group):
                K = np.asarray(group["K"])
                R = np.asarray(group["R"])
                t = np.asarray(group["T"] if "T" in group else group["t"])
                return K.reshape(3, 3), R.reshape(3, 3), t.reshape(3)

    raise KeyError(f"K/R/T not found in {h5_path}")


def load_cam_from_dir(calib_dir: Path, img_path: Path, flip_w2c: bool) -> CameraParams:
    K, R, t = _read_cam_from_h5(calib_dir / f"calibration_{img_path.stem}.h5")
    if flip_w2c:
        R, t = R.T, -R.T @ t
    return CameraParams(K=K, R=R, t=t)


def relative_pose(cam_a: CameraParams, cam_b: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
    R_ba = cam_b.R @ cam_a.R.T
    t_ba = cam_b.t - R_ba @ cam_a.t
    return R_ba, t_ba


def symmetric_epi_errors(
    x0_px: np.ndarray,
    x1_px: np.ndarray,
    cam0: CameraParams,
    cam1: CameraParams,
    device: torch.device,
) -> np.ndarray:
    if x0_px.size == 0:
        return np.zeros((0,), dtype=np.float64)

    R01, t01 = relative_pose(cam0, cam1)
    tx = np.array(
        [
            [0.0, -t01[2], t01[1]],
            [t01[2], 0.0, -t01[0]],
            [-t01[1], t01[0], 0.0],
        ],
        dtype=np.float32,
    )
    E = tx @ R01

    pts0 = torch.from_numpy(x0_px.astype(np.float32)).to(device)
    pts1 = torch.from_numpy(x1_px.astype(np.float32)).to(device)
    E_t = torch.from_numpy(E.astype(np.float32)).to(device)
    K0_t = torch.from_numpy(cam0.K.astype(np.float32)).to(device)
    K1_t = torch.from_numpy(cam1.K.astype(np.float32)).to(device)

    with torch.no_grad():
        errs = symmetric_epipolar_distance(pts0, pts1, E_t, K0_t, K1_t)
    return errs.detach().cpu().numpy().astype(np.float64)
