#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build bag-wise IMC custom multiview inputs from PhotoTourism / MegaDepth bag5 subsets
using sequential SuperPoint/SIFT + LightGlue matching on edges:
  0-1, 1-2, 2-3, 3-4

Only tracks connected from the first frame to the last frame are kept.

Outputs:
  <out_root>/<method_name>/<dataset_name>/<scene>/<bag_name>/keypoints.h5
  <out_root>/<method_name>/<dataset_name>/<scene>/<bag_name>/descriptors.h5
  <out_root>/<method_name>/<dataset_name>/<scene>/<bag_name>/matches_multiview.h5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn

from lightglue import LightGlue, SuperPoint, SIFT
from lightglue.utils import load_image, rbd
from thop import profile


# ============================================================
# Helpers
# ============================================================

def read_bag_paths(bag_file: Path) -> List[str]:
    return [ln.strip() for ln in bag_file.read_text(encoding="utf-8").splitlines() if ln.strip()]


def as_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ============================================================
# Track building
# ============================================================

def build_tracks_from_sequential_pairs(
    pair_matches: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    topk: int = 20000,
    method: str = "default",
    nn_max_d2: float = 50.0,
) -> Dict[int, dict]:
    tracks: Dict[int, dict] = {}
    next_tid = 0

    prev_points: Optional[np.ndarray] = None
    prev_tids: Optional[np.ndarray] = None
    prev_point_to_tid: Dict[tuple, int] = {}

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
            n_cur = int(mk0.shape[0])
            if n_cur == 0:
                prev_points = None
                prev_tids = None
                continue

            curr_points_list = []
            curr_tids_list = []

            if prev_points is None or prev_points.shape[0] == 0:
                for j in range(n_cur):
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

            used_current = np.zeros(n_cur, dtype=bool)

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

            for k_cur in range(n_cur):
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

        else:
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
        start_id = int(tr["start_id"])
        end_id = int(tr["end_id"])
        pts = tr["points"]
        if start_id == 0 and end_id == bag_size - 1 and len(pts) == bag_size:
            out[tid] = tr
    return out


# ============================================================
# Per-bag image keypoint registry
# ============================================================

class ImageKeypointRegistry:
    def __init__(self, quant: float = 0.5):
        self.quant = float(quant)
        self.points_by_image: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.index_by_image: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(dict)

    def _qkey(self, xy: np.ndarray) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        return (int(round(x / self.quant)), int(round(y / self.quant)))

    def get_or_add(self, image_name: str, xy: np.ndarray) -> int:
        qk = self._qkey(xy)
        if qk in self.index_by_image[image_name]:
            return self.index_by_image[image_name][qk]
        idx = len(self.points_by_image[image_name])
        self.points_by_image[image_name].append((float(xy[0]), float(xy[1])))
        self.index_by_image[image_name][qk] = idx
        return idx

    def as_h5_dict(self) -> Dict[str, np.ndarray]:
        out = {}
        for name, pts in self.points_by_image.items():
            if len(pts) == 0:
                out[name] = np.zeros((0, 2), dtype=np.float32)
            else:
                out[name] = np.asarray(pts, dtype=np.float32)
        return out


# ============================================================
# Convert tracks -> all-pair matches
# ============================================================

def induce_pairwise_matches_from_tracks(
    bag_rel_paths: List[str],
    tracks: Dict[int, dict],
    registry: ImageKeypointRegistry,
    min_track_len: int = 2,
) -> Dict[str, set]:
    pair_matches = defaultdict(set)

    for _, tr in tracks.items():
        pts = np.asarray(tr["points"], dtype=np.float32)
        start_id = int(tr["start_id"])

        if len(pts) < min_track_len:
            continue

        obs = []
        seen_frames = set()
        for j in range(len(pts)):
            frame_id = start_id + j
            if frame_id < 0 or frame_id >= len(bag_rel_paths):
                continue
            if frame_id in seen_frames:
                continue
            seen_frames.add(frame_id)

            img_name = Path(bag_rel_paths[frame_id]).name
            kp_idx = registry.get_or_add(img_name, pts[j])
            obs.append((img_name, kp_idx))

        if len(obs) < 2:
            continue

        for a in range(len(obs)):
            for b in range(a + 1, len(obs)):
                imgA, idxA = obs[a]
                imgB, idxB = obs[b]
                if imgA < imgB:
                    pk = f"{imgA}-{imgB}"
                    pair_matches[pk].add((idxA, idxB))
                else:
                    pk = f"{imgB}-{imgA}"
                    pair_matches[pk].add((idxB, idxA))

    return pair_matches


# ============================================================
# Prune keypoints and reindex matches
# ============================================================

def prune_keypoints_and_matches(
    kp_dict: Dict[str, np.ndarray],
    pair_dict: Dict[str, set],
    max_kp_per_image: int = 2048,
):
    usage = {img: np.zeros(len(kps), dtype=np.int64) for img, kps in kp_dict.items()}

    for pair_name, pairs in pair_dict.items():
        imgA, imgB = pair_name.split("-")
        for ia, ib in pairs:
            if imgA in usage and 0 <= ia < len(usage[imgA]):
                usage[imgA][ia] += 1
            if imgB in usage and 0 <= ib < len(usage[imgB]):
                usage[imgB][ib] += 1

    keep_maps = {}
    new_kp_dict = {}

    for img, kps in kp_dict.items():
        n = len(kps)
        if n <= max_kp_per_image:
            keep_idx = np.arange(n, dtype=np.int32)
        else:
            freq = usage[img]
            order = np.argsort(-freq)
            keep_idx = np.sort(order[:max_kp_per_image]).astype(np.int32)

        old_to_new = {int(old): int(new) for new, old in enumerate(keep_idx.tolist())}
        keep_maps[img] = old_to_new
        new_kp_dict[img] = kps[keep_idx]

    new_pair_dict = defaultdict(set)

    for pair_name, pairs in pair_dict.items():
        imgA, imgB = pair_name.split("-")
        mapA = keep_maps.get(imgA, {})
        mapB = keep_maps.get(imgB, {})

        for ia, ib in pairs:
            if ia in mapA and ib in mapB:
                new_pair_dict[pair_name].add((mapA[ia], mapB[ib]))

    return new_kp_dict, new_pair_dict


# ============================================================
# HDF5 writers
# ============================================================

def write_keypoints_h5(path: Path, kp_dict: Dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, arr in sorted(kp_dict.items()):
            f.create_dataset(name, data=np.asarray(arr, dtype=np.float32))


def write_dummy_descriptors_h5(path: Path, kp_dict: Dict[str, np.ndarray], dim: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, arr in sorted(kp_dict.items()):
            n = int(arr.shape[0])
            f.create_dataset(name, data=np.zeros((n, dim), dtype=np.float32))


def write_matches_h5(path: Path, pair_dict: Dict[str, set]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, pairs in sorted(pair_dict.items()):
            if len(pairs) == 0:
                arr = np.zeros((0, 2), dtype=np.int32)
            else:
                arr = np.asarray(sorted(pairs), dtype=np.int32)
            f.create_dataset(name, data=arr)


# ============================================================
# SP/LG wrappers
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


@torch.no_grad()
def run_splg_pair(
    device: torch.device,
    imgA: Path,
    imgB: Path,
    prev_data: Any = None,
    sp_model: Any = None,
    lg_model: Any = None,
    image_idA: int = 0,
    image_idB: int = 1,
):
    if sp_model is None or lg_model is None:
        raise ValueError("sp_model と lg_model を渡してください")

    image0 = load_image(str(imgA)).to(device)
    image1 = load_image(str(imgB)).to(device)
    if image0 is None or image1 is None:
        raise FileNotFoundError(f"Failed to read {imgA} or {imgB}")

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        t0 = time.perf_counter()

    feats0_b = sp_model.extract(image0)
    feats1_b = sp_model.extract(image1)
    matches01_b = lg_model({"image0": feats0_b, "image1": feats1_b})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0_b, feats1_b, matches01_b]]

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        runtime_ms = float(start_event.elapsed_time(end_event))
    else:
        runtime_ms = float((time.perf_counter() - t0) * 1000.0)

    matches = matches01.get("matches", None)
    if matches is None or matches.numel() == 0:
        mk0 = torch.empty((0, 2), device=device, dtype=torch.float32)
        mk1 = torch.empty((0, 2), device=device, dtype=torch.float32)
        mconf = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        points0 = feats0["keypoints"][matches[:, 0]].float()
        points1 = feats1["keypoints"][matches[:, 1]].float()
        mconf = matches01.get("scores", torch.ones(points0.shape[0], device=device)).float()
        mk0 = points0
        mk1 = points1

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

    result = {}
    return mk0, mk1, mconf, flops, runtime_ms, result


def build_splg_models(
    device: torch.device,
    features: str = "superpoint",
    sp_max_keypoints: int = -1,
    sp_det_thr: float = 0.0,
    lg_depth_conf: float = -1.0,
    lg_width_conf: float = -1.0,
):
    max_k = None if sp_max_keypoints < 0 else int(sp_max_keypoints)
    if features == "superpoint":
        extractor = SuperPoint(
            max_num_keypoints=max_k,
            detection_threshold=float(sp_det_thr),
        ).eval().to(device)
        matcher = LightGlue(
            features="superpoint",
            depth_confidence=float(lg_depth_conf),
            width_confidence=float(lg_width_conf),
        ).eval().to(device)
    elif features == "sift":
        extractor = SIFT(
            max_num_keypoints=max_k,
            detection_threshold=float(sp_det_thr),
        ).eval().to(device)
        matcher = LightGlue(
            features="sift",
            depth_confidence=float(lg_depth_conf),
            width_confidence=float(lg_width_conf),
        ).eval().to(device)
    else:
        raise ValueError(f"Unknown features: {features}")
    return extractor, matcher


# ============================================================
# Main scene processing
# ============================================================

def process_scene(
    scene_root: Path,
    dataset_root: Path,
    subset_dir: Path,
    out_root: Path,
    device: torch.device,
    method: str,
    method_name: str,
    bag_size: int,
    topk: int,
    nn_max_d2: float,
    min_track_len: int,
    quant: float,
    max_kp_per_image: int,
    min_full_tracks: int,
    sp_model: Any,
    lg_model: Any,
    dataset_name: str,
):
    bag_files = sorted(subset_dir.glob(f"{bag_size}bag_*.txt"))
    if not bag_files:
        raise FileNotFoundError(f"No {bag_size}bag_*.txt in {subset_dir}")

    scene_name = scene_root.name
    print(f"[scene] {scene_name}  bags={len(bag_files)}")

    scene_stats = []

    for bag_file in bag_files:
        rel_paths = read_bag_paths(bag_file)
        bag_name = bag_file.stem

        if len(rel_paths) != bag_size:
            print(f"[skip] {bag_file.name}: expected {bag_size} images, got {len(rel_paths)}")
            continue

        img_paths = [dataset_root / rp for rp in rel_paths]
        for p in img_paths:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")

        pair_matches = []
        image_idA = 0
        image_idB = 1

        for i in range(bag_size - 1):
            mk0_t, mk1_t, mconf_t, _, _, _ = run_splg_pair(
                device=device,
                imgA=img_paths[i],
                imgB=img_paths[i + 1],
                prev_data=None,
                sp_model=sp_model,
                lg_model=lg_model,
                image_idA=image_idA,
                image_idB=image_idB,
            )
            image_idA += 1
            image_idB += 1

            mk0 = as_np(mk0_t).astype(np.float32)
            mk1 = as_np(mk1_t).astype(np.float32)
            mconf = as_np(mconf_t).astype(np.float32) if mconf_t is not None else None
            pair_matches.append((mk0, mk1, mconf))

        tracks = build_tracks_from_sequential_pairs(
            pair_matches=pair_matches,
            topk=topk,
            method="default",
            nn_max_d2=nn_max_d2,
        )

        all_lengths = [len(t["points"]) for t in tracks.values()]
        full_tracks = filter_full_length_tracks(tracks, bag_size=bag_size)
        full_lengths = [len(t["points"]) for t in full_tracks.values()]

        if len(full_tracks) < min_full_tracks:
            print(f"  [skip] {bag_name}: full_tracks={len(full_tracks)} < min_full_tracks={min_full_tracks}")
            continue

        registry = ImageKeypointRegistry(quant=quant)
        pairwise = induce_pairwise_matches_from_tracks(
            bag_rel_paths=rel_paths,
            tracks=full_tracks,
            registry=registry,
            min_track_len=min_track_len,
        )

        kp_dict = registry.as_h5_dict()
        kp_dict, pairwise = prune_keypoints_and_matches(
            kp_dict=kp_dict,
            pair_dict=pairwise,
            max_kp_per_image=max_kp_per_image,
        )

        bag_out = out_root / method_name / dataset_name / scene_name / bag_name
        write_keypoints_h5(bag_out / "keypoints.h5", kp_dict)
        write_dummy_descriptors_h5(bag_out / "descriptors.h5", kp_dict, dim=1)
        write_matches_h5(bag_out / "matches_multiview.h5", pairwise)

        meta = {
            "method": method_name,
            "scene": scene_name,
            "bag": bag_name,
            "bag_file": str(bag_file),
            "num_images": int(len(kp_dict)),
            "num_pair_keys": int(len(pairwise)),
            "max_kp_per_image": int(max_kp_per_image),
            "num_tracks_all": int(len(tracks)),
            "num_tracks_full_length": int(len(full_tracks)),
            "track_len_all_max": int(max(all_lengths) if all_lengths else 0),
            "track_len_full_max": int(max(full_lengths) if full_lengths else 0),
        }
        (bag_out / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        scene_stats.append(meta)

        print(
            f"  [bag] {bag_name} "
            f"all_tracks={len(tracks)} "
            f"full_tracks={len(full_tracks)} "
            f"pair_keys={len(pairwise)} "
            f"out={bag_out}"
        )

    scene_meta = {
        "method": method_name,
        "scene": scene_name,
        "num_bags_written": len(scene_stats),
        "bags": scene_stats,
    }
    scene_meta_path = out_root / method_name / dataset_name / scene_name / "scene_meta.json"
    scene_meta_path.parent.mkdir(parents=True, exist_ok=True)
    scene_meta_path.write_text(json.dumps(scene_meta, indent=2), encoding="utf-8")

    print(f"[done] {scene_name}")
    print(f"  scene meta: {scene_meta_path}")


# ============================================================
# Defaults / Args
# ============================================================

use_dataset = "IMC"  # IMC or megadepth

if use_dataset == "IMC":
    default_root = Path("/home/ach17765lb/data/phototourism")
    dataset_name = "phototourism"
    scene = "all"
    set_name = "set_100"
    subset_dir = default_root / scene / set_name / "sub_set"
    dataset_root = default_root / scene / set_name
    DEFAULT_OUT_ROOT = Path("/home/ach17765lb/imc_features_bagwise")
    ALL_PT_SCENES = ["reichstag", "sacre_coeur", "st_peters_square"]
else:
    default_root = Path("/home/ach17765lb/JamMa/data/megadepth/Undistorted_SfM/")
    dataset_name = "megadepth"
    scene = "all"
    subset_dir = default_root / scene / "5bag"
    dataset_root = Path("/home/ach17765lb/JamMa/data/megadepth/")
    DEFAULT_OUT_ROOT = Path("/home/ach17765lb/megadepth_features_bagwise")
    ALL_PT_SCENES = ["0015", "0022"]






def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=use_dataset, choices=["IMC", "megadepth"])
    ap.add_argument("--phototourism_root", type=Path, default=Path("/home/ach17765lb/data/phototourism"))
    ap.add_argument("--megadepth_undist_root", type=Path, default=Path("/home/ach17765lb/JamMa/data/megadepth/Undistorted_SfM/"))
    ap.add_argument("--megadepth_root", type=Path, default=Path("/home/ach17765lb/JamMa/data/megadepth/"))
    ap.add_argument("--scene", type=str, default=scene)
    ap.add_argument("--set_name", type=str, default="set_100")
    ap.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--method", type=str, default="splg_sp", choices=["splg_sp", "splg_sift"])
    ap.add_argument("--method_name", type=str, default=None, help="Output method name. If omitted, use --method.")
    ap.add_argument("--bag_size", type=int, default=5)
    ap.add_argument("--topk", type=int, default=20000)
    ap.add_argument("--nn_max_d2", type=float, default=50.0)
    ap.add_argument("--min_track_len", type=int, default=2)
    ap.add_argument("--quant", type=float, default=0.5)
    ap.add_argument("--max_kp_per_image", type=int, default=20000)
    ap.add_argument("--min_full_tracks", type=int, default=1)

    ap.add_argument("--sp_max_keypoints", type=int, default=-1)
    ap.add_argument("--sp_det_thr", type=float, default=0.0)
    ap.add_argument("--lg_depth_conf", type=float, default=-1.0)
    ap.add_argument("--lg_width_conf", type=float, default=-1.0)

    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    method_name = args.method_name if args.method_name is not None else args.method

    print(f"[device] {device}")
    print(f"[method_name] {method_name}")

    feats = "superpoint" if args.method == "splg_sp" else "sift"
    sp_model, lg_model = build_splg_models(
        device=device,
        features=feats,
        sp_max_keypoints=args.sp_max_keypoints,
        sp_det_thr=args.sp_det_thr,
        lg_depth_conf=args.lg_depth_conf,
        lg_width_conf=args.lg_width_conf,
    )
    print(f"[matcher] {feats}+LightGlue")

    if args.dataset == "IMC":
        dataset_name_local = "phototourism"
        scenes = ALL_PT_SCENES if args.scene == "all" else [args.scene]
        for scene_name in scenes:
            scene_root = args.phototourism_root / scene_name
            if not scene_root.exists():
                raise FileNotFoundError(scene_root)

            dataset_root_local = scene_root / args.set_name
            subset_dir_local = dataset_root_local / "sub_set"

            process_scene(
                scene_root=scene_root,
                dataset_root=dataset_root_local,
                subset_dir=subset_dir_local,
                out_root=args.out_root,
                device=device,
                method=args.method,
                method_name=method_name,
                bag_size=args.bag_size,
                topk=args.topk,
                nn_max_d2=args.nn_max_d2,
                min_track_len=args.min_track_len,
                quant=args.quant,
                max_kp_per_image=args.max_kp_per_image,
                min_full_tracks=args.min_full_tracks,
                sp_model=sp_model,
                lg_model=lg_model,
                dataset_name=dataset_name_local,
            )
    else:
        dataset_name_local = "megadepth"
        scenes = ALL_PT_SCENES if args.scene == "all" else [args.scene]
        for scene_name in scenes:
            scene_root = args.megadepth_undist_root / scene_name
            if not scene_root.exists():
                raise FileNotFoundError(scene_root)

            subset_dir_local = scene_root / f"{args.bag_size}bag"
            if not subset_dir_local.exists():
                subset_dir_local = scene_root / "5bag"

            process_scene(
                scene_root=scene_root,
                dataset_root=args.megadepth_root,
                subset_dir=subset_dir_local,
                out_root=args.out_root,
                device=device,
                method=args.method,
                method_name=method_name,
                bag_size=args.bag_size,
                topk=args.topk,
                nn_max_d2=args.nn_max_d2,
                min_track_len=args.min_track_len,
                quant=args.quant,
                max_kp_per_image=args.max_kp_per_image,
                min_full_tracks=args.min_full_tracks,
                sp_model=sp_model,
                lg_model=lg_model,
                dataset_name=dataset_name_local,
            )


if __name__ == "__main__":
    main()