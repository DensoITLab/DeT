#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate JamMa's local shift-equivariance assumption on IMC 3-image bags.

For each 3-bag, run default JamMa on adjacent pairs (0-1 and 1-2).
For every matched point x_a in image A, generate perturbed query points
x_a' = x_a + delta, where ||delta|| <= r and r in {0,1,2,3} px.

Using depth and camera geometry, compute the GT projection y_gt of x_a'
in image B. Then compare it with y_pred = y_match + delta, where y_match
is JamMa's original matched point in image B.

This tests whether the local displacement on image A can be transferred to
image B, i.e. whether the matcher behaves approximately equivariantly under
small shifts.

Outputs:
  - JSON summary with error histograms per radius
  - NPZ with raw errors and displacement magnitudes

This script is intentionally aligned with your existing evaluation script:
  - uses read_megadepth_color
  - uses PL_JamMa
  - uses calibration_<stem>.h5 and <stem>.h5 depth files
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.utils.dataset import read_megadepth_color
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler

# --- Dataset defaults (PhotoTourism example) ---
default_root = Path('/home/ach17765lb/data/phototourism')
scene = 'reichstag'   # 'reichstag' , 'sacre_coeur', 'st_peters_square'
set_name = 'set_100'
subset_size = 3


@dataclasses.dataclass
class CameraParams:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_cfg_path', type=str, default='configs/data/megadepth_test_1500.py')
    parser.add_argument('--main_cfg_path', type=str, default='configs/jamma/outdoor/test.py')
    parser.add_argument('--ckpt_path', type=str, default='official')
    parser.add_argument('--dump_dir', type=str, default='dump/jamma_outdoor')
    parser.add_argument('--profiler_name', type=str, default='inference')

    parser.add_argument('--subset_dir', type=Path, default=default_root / scene / set_name / 'sub_set',
                        help='Contains Nbag_*.txt files (e.g., 5bag_015.txt)')
    parser.add_argument('--dataset_root', type=Path, default=default_root / scene / set_name,
                        help='Root to prepend to image relative paths in bag files')
    parser.add_argument('--calib_dir', type=Path, default=default_root / scene / set_name / 'calibration',
                        help='Directory with calibration_<stem>.h5 per image')
    parser.add_argument('--depth_dir', type=Path, default=default_root / scene / set_name / 'depth_maps',
                        help='Directory with <stem>.h5 per image (depth)')

    parser.add_argument('--bag_size', type=int, default=subset_size, help='Number of images in each bag file')
    parser.add_argument('--flip_w2c', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--radii', type=float, nargs='+', default=[0.0, 1.0, 3.0, 4.47, 5.0, 10.0],
                        help='Perturbation radii in pixels')
    parser.add_argument('--samples_per_point_per_radius', type=int, default=1,
                        help='Number of random perturbations per matched point and radius')
    parser.add_argument('--topk', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--hist_bins', type=float, nargs='+',
                        default=[0, 0.5, 1, 2, 3, 5, 10, 20, 50, 100],
                        help='Histogram bin edges for reprojection error in pixels')

    parser.add_argument('--save_json', type=Path, default=default_root / 'results_imc_eval_jamma_hist.json',
                        help='Path to save JSON results')
    parser.add_argument('--save_npz', type=Path, default=None)

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def read_bag_paths(bag_file: Path) -> List[str]:
    return [ln.strip() for ln in bag_file.read_text(encoding='utf-8').splitlines() if ln.strip()]


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
                raise KeyError(f'K/R/T not found in {h5_path}')
    return K.reshape(3, 3), R.reshape(3, 3), T.reshape(3)


def load_cam_from_dir(calib_dir: Path, img_path: Path, flip_w2c: bool) -> CameraParams:
    h5_path = calib_dir / f'calibration_{img_path.stem}.h5'
    K, R, t = _read_cam_from_h5(h5_path)
    if flip_w2c:
        R, t = R.T, -R.T @ t
    return CameraParams(K.astype(np.float32), R.astype(np.float32), t.astype(np.float32))


def _read_depth_from_h5(h5_path: Path) -> np.ndarray:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, 'r') as f:
        if 'depth' in f:
            depth = np.array(f['depth'])
        else:
            depth = None
            for key in f.keys():
                arr = np.array(f[key])
                if arr.ndim == 2:
                    depth = arr
                    break
            if depth is None:
                raise KeyError(f'depth dataset not found in {h5_path}')
    return depth.astype(np.float32)


def load_depth_from_dir(depth_dir: Path, img_path: Path) -> np.ndarray:
    return _read_depth_from_h5(depth_dir / f'{img_path.stem}.h5')


def run_jamma_pair(device: torch.device, img_a: Path, img_b: Path, jamma: PL_JamMa):
    image0, s0, m0, p0, *_ = read_megadepth_color(str(img_a), 832, 8, True)
    image1, s1, m1, p1, *_ = read_megadepth_color(str(img_b), 832, 8, True)

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
        image_idA=0,
        image_idB=1,
    )

    jamma = jamma.to(device).eval()
    with torch.no_grad():
        result, _, _ = jamma(data)

    mk0 = result['mkpts0_f_origin']
    mk1 = result['mkpts1_f_origin']
    mconf = result.get('mconf_f', None)
    return mk0, mk1, mconf,  result


def project_with_depth(
    p_src: np.ndarray,
    depth_src: np.ndarray,
    cam_src: CameraParams,
    cam_tgt: CameraParams,
    tgt_shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], bool]:
    """Project one source pixel into target image using source depth."""
    h_src, w_src = depth_src.shape
    x, y = float(p_src[0]), float(p_src[1])
    ix, iy = int(round(x)), int(round(y))

    if ix < 0 or ix >= w_src or iy < 0 or iy >= h_src:
        return None, False

    z = float(depth_src[iy, ix])
    if not np.isfinite(z) or z <= 0:
        return None, False

    k_inv = np.linalg.inv(cam_src.K)
    p_h = np.array([x, y, 1.0], dtype=np.float32)
    x_cam_src = k_inv @ (p_h * z)
    x_world = cam_src.R.T @ (x_cam_src - cam_src.t.reshape(3))
    x_cam_tgt = cam_tgt.R @ x_world + cam_tgt.t.reshape(3)

    if x_cam_tgt[2] <= 0:
        return None, False

    p_tgt_h = cam_tgt.K @ (x_cam_tgt / x_cam_tgt[2])
    p_tgt = np.array([float(p_tgt_h[0]), float(p_tgt_h[1])], dtype=np.float32)

    h_tgt, w_tgt = tgt_shape
    if p_tgt[0] < 0 or p_tgt[0] >= w_tgt or p_tgt[1] < 0 or p_tgt[1] >= h_tgt:
        return None, False

    return p_tgt, True


def sample_delta_at_radius(rng: np.random.Generator, radius: float) -> np.ndarray:
    """Sample a random 2D vector with exactly the specified radius."""
    if radius == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float32)


def evaluate_pair(
    img_a: Path,
    img_b: Path,
    cam_a: CameraParams,
    cam_b: CameraParams,
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    jamma: PL_JamMa,
    device: torch.device,
    radii: List[float],
    samples_per_point_per_radius: int,
    topk: int,
    rng: np.random.Generator,
) -> Dict:
    mk_a_t, mk_b_t, mconf_t, result = run_jamma_pair(device, img_a, img_b, jamma)


    scale0 = result["scale0"][0].detach().cpu().numpy()  # shape: (2,)
    scale1 = result["scale1"][0].detach().cpu().numpy()  # shape: (2,)

    if mconf_t is not None and mconf_t.numel() > 0:
        k = min(int(mconf_t.numel()), topk)
        idx = torch.topk(mconf_t, k, 0).indices
        mk_a = mk_a_t[idx].detach().cpu().numpy().astype(np.float32)
        mk_b = mk_b_t[idx].detach().cpu().numpy().astype(np.float32)
        conf = mconf_t[idx].detach().cpu().numpy().astype(np.float32)
    else:
        mk_a = mk_a_t.detach().cpu().numpy().astype(np.float32)
        mk_b = mk_b_t.detach().cpu().numpy().astype(np.float32)
        conf = np.ones((mk_a.shape[0],), dtype=np.float32)

    h_a, w_a = depth_a.shape
    h_b, w_b = depth_b.shape
    print(f'[evaluate_pair] {img_a.name} → {img_b.name}: {mk_a.shape[0]} matches')

    pair_out = {
        'img_a': str(img_a),
        'img_b': str(img_b),
        'num_matches': int(mk_a.shape[0]),
        'radii': {},
    }

    for radius in radii:
        errors = []
        delta_norms = []
        valid_flags = []
        confs = []

        for p_a, p_b, c in zip(mk_a, mk_b, conf):
            for _ in range(samples_per_point_per_radius):
                delta = sample_delta_at_radius(rng, float(radius))
                p_a_shift = (p_a / scale0 + delta) * scale0

                if p_a_shift[0] < 0 or p_a_shift[0] >= w_a or p_a_shift[1] < 0 or p_a_shift[1] >= h_a:
                    valid_flags.append(False)
                    continue

                p_b_gt, ok = project_with_depth(
                    p_src=p_a_shift,
                    depth_src=depth_a,
                    cam_src=cam_a,
                    cam_tgt=cam_b,
                    tgt_shape=(h_b, w_b),
                )
                if not ok:
                    valid_flags.append(False)
                    continue

                # Core assumption under test:
                # If the source-side point moves by delta, the target-side match moves by the same delta.
                p_b_pred = p_b / scale1 + delta
                err = np.max(np.abs(p_b_pred - p_b_gt / scale1), axis=-1)  # use max-norm error in pixels
                print(f'  radius={radius:.2f} px, p_a={p_a}, delta={delta}, p_b_gt={p_b_gt}, p_b_pred={p_b_pred}, err={err:.2f} px')

                errors.append(float(err))
                delta_norms.append(float(np.linalg.norm(delta)))
                confs.append(float(c))
                valid_flags.append(True)

        errors_np = np.asarray(errors, dtype=np.float32)
        pair_out['radii'][str(radius)] = {
            'n_valid': int(errors_np.size),
            'n_attempted': int(mk_a.shape[0] * samples_per_point_per_radius),
            'valid_ratio': float(errors_np.size / max(1, mk_a.shape[0] * samples_per_point_per_radius)),
            'mean_err_px': float(np.mean(errors_np)) if errors_np.size else None,
            'median_err_px': float(np.median(errors_np)) if errors_np.size else None,
            'p90_err_px': float(np.percentile(errors_np, 90)) if errors_np.size else None,
            'p95_err_px': float(np.percentile(errors_np, 95)) if errors_np.size else None,
            'errors_px': errors,
            'delta_norms_px': delta_norms,
            'conf': confs,
        }

    return pair_out


def summarize(all_pair_results: List[Dict], hist_bins: List[float]) -> Dict:
    summary = {
        'hist_bins_px': hist_bins,
        'by_radius': {},
    }

    all_radii = sorted({r for p in all_pair_results for r in p['radii'].keys()}, key=lambda x: float(x))
    bins_np = np.asarray(hist_bins, dtype=np.float32)

    for r in all_radii:
        errs = []
        n_attempted = 0
        n_valid = 0
        for p in all_pair_results:
            rec = p['radii'][r]
            errs.extend(rec['errors_px'])
            n_attempted += int(rec['n_attempted'])
            n_valid += int(rec['n_valid'])

        errs_np = np.asarray(errs, dtype=np.float32)
        hist, _ = np.histogram(errs_np, bins=bins_np) if errs_np.size else (np.zeros(len(bins_np) - 1, dtype=int), bins_np)

        summary['by_radius'][r] = {
            'n_attempted': int(n_attempted),
            'n_valid': int(n_valid),
            'valid_ratio': float(n_valid / max(1, n_attempted)),
            'mean_err_px': float(np.mean(errs_np)) if errs_np.size else None,
            'median_err_px': float(np.median(errs_np)) if errs_np.size else None,
            'p90_err_px': float(np.percentile(errs_np, 90)) if errs_np.size else None,
            'p95_err_px': float(np.percentile(errs_np, 95)) if errs_np.size else None,
            'ratio_le_1px': float(np.mean(errs_np <= 1.0)) if errs_np.size else 0.0,
            'ratio_le_3px': float(np.mean(errs_np <= 3.0)) if errs_np.size else 0.0,
            'ratio_le_5px': float(np.mean(errs_np <= 5.0)) if errs_np.size else 0.0,
            'ratio_le_10px': float(np.mean(errs_np <= 10.0)) if errs_np.size else 0.0,
            'hist_counts': hist.astype(int).tolist(),
        }

    return summary


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    # Default JamMa, not DeT.
    config.JAMMA.DET.USE_DET = False
    config.JAMMA.USE_COMPILE = False

    profiler = build_profiler(args.profiler_name)
    jamma = PL_JamMa(config, pretrained_ckpt=args.ckpt_path,
                     profiler=profiler, dump_dir=args.dump_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    jamma = jamma.to(device).eval()

    bag_files = sorted(args.subset_dir.glob(f'{args.bag_size}bag_*.txt'))
    if not bag_files:
        raise FileNotFoundError(f'No {args.bag_size}bag_*.txt found in {args.subset_dir}')

    all_pair_results = []

    for bag_file in bag_files:
        rel_paths = read_bag_paths(bag_file)
        if len(rel_paths) != args.bag_size:
            print(f'[SKIP] {bag_file}: expected {args.bag_size} images, got {len(rel_paths)}')
            continue

        img_paths = [args.dataset_root / rp for rp in rel_paths]
        cams = [load_cam_from_dir(args.calib_dir, p, args.flip_w2c) for p in img_paths]
        depths = [load_depth_from_dir(args.depth_dir, p) for p in img_paths]

        # IMC 3-image bag: adjacent pairs 0-1 and 1-2.
        for i in range(args.bag_size - 1):
            print(f'[Bag {bag_file.name}] Pair {i}-{i+1}: {img_paths[i].name} -> {img_paths[i+1].name}')
            pair_result = evaluate_pair(
                img_a=img_paths[i],
                img_b=img_paths[i + 1],
                cam_a=cams[i],
                cam_b=cams[i + 1],
                depth_a=depths[i],
                depth_b=depths[i + 1],
                jamma=jamma,
                device=device,
                radii=args.radii,
                samples_per_point_per_radius=args.samples_per_point_per_radius,
                topk=args.topk,
                rng=rng,
            )
            pair_result['bag_file'] = str(bag_file)
            pair_result['pair_index'] = i
            all_pair_results.append(pair_result)

    summary = summarize(all_pair_results, args.hist_bins)
    out = {
        'config': {
            'bag_size': args.bag_size,
            'radii': args.radii,
            'samples_per_point_per_radius': args.samples_per_point_per_radius,
            'topk': args.topk,
            'hist_bins': args.hist_bins,
            'definition': 'p_b_pred = p_b_match + delta, error = ||p_b_pred - GT_depth_projection(p_a_match + delta)||_2',
        },
        'summary': summary,
        'per_pair': all_pair_results,
    }

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'[OK] saved JSON: {args.save_json}')

    if args.save_npz is not None:
        flat = {}
        for r in summary['by_radius'].keys():
            errs = []
            for p in all_pair_results:
                errs.extend(p['radii'][r]['errors_px'])
            flat[f'errors_radius_{r}'] = np.asarray(errs, dtype=np.float32)
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_npz, **flat)
        print(f'[OK] saved NPZ: {args.save_npz}')


if __name__ == '__main__':
    main()
