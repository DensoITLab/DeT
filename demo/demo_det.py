#!/usr/bin/env python3
import argparse
import colorsys
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_IMAGE_PATHS = [
    PROJECT_ROOT / "demo" / "aseets" / "sample1.jpg",
    PROJECT_ROOT / "demo" / "aseets" / "sample2.jpg",
    PROJECT_ROOT / "demo" / "aseets" / "sample3.jpg",
]
DEFAULT_CKPT_PATH = "weights/jamma.ckpt"

from src.config.default import get_cfg_defaults
from src.jamma.backbone import CovNextV2_nano
from src.jamma.jamma import JamMa


class DeTMatcher(torch.nn.Module):
    def __init__(self, jamma_config: dict, ckpt_path: str):
        super().__init__()
        self.backbone = CovNextV2_nano()
        self.matcher = JamMa(jamma_config)

        if ckpt_path == "official":
            state_dict = torch.hub.load_state_dict_from_url(
                "https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt",
                file_name="jamma.ckpt",
            )["state_dict"]
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

        self.load_state_dict(state_dict, strict=True)

    def forward(self, data: dict) -> dict:
        self.backbone(data)
        self.matcher(data, mode="test")
        return data


def resolve_input_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def lower_config(yacs_cfg):
    if not hasattr(yacs_cfg, "items"):
        return yacs_cfg
    return {key.lower(): lower_config(value) for key, value in yacs_cfg.items()}


def get_resized_wh(width: int, height: int, resize: Optional[int]):
    if resize is None:
        return width, height
    scale = resize / max(width, height)
    return int(round(width * scale)), int(round(height * scale))


def get_divisible_wh(width: int, height: int, divisor: Optional[int]):
    if divisor is None:
        return width, height
    return int(width // divisor * divisor), int(height // divisor * divisor)


def pad_bottom_right(image: np.ndarray, pad_size: int):
    padded = np.zeros((image.shape[0], pad_size, pad_size), dtype=image.dtype)
    padded[:, :image.shape[1], :image.shape[2]] = image

    mask = np.zeros((pad_size, pad_size), dtype=bool)
    mask[:image.shape[1], :image.shape[2]] = True
    return padded, mask


def read_color(path: Path, resize: Optional[int], divisor: int, padding: bool):
    image = Image.open(path).convert("RGB")
    width, height = image.width, image.height
    new_width, new_height = get_resized_wh(width, height, resize)
    new_width, new_height = get_divisible_wh(new_width, new_height, divisor)

    scale = torch.tensor([width / new_width, height / new_height], dtype=torch.float)
    image = image.resize((new_width, new_height), Image.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_np = (image_np - mean) / std

    if padding:
        image_np, mask_np = pad_bottom_right(image_np, max(new_height, new_width))
        mask = torch.from_numpy(mask_np)
    else:
        mask = None

    image_tensor = torch.from_numpy(image_np).float()[None]
    prepad_size = torch.tensor([new_height, new_width])
    return image_tensor, scale, mask, prepad_size


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeT/JamMa tracking on a sequence of three or more images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Ordered image paths. Defaults to demo/aseets/sample1.jpg sample2.jpg sample3.jpg.",
    )
    parser.add_argument("--image_dir", type=Path, default=None, help="Directory with ordered images.")
    parser.add_argument("--pattern", type=str, default="*.jpg", help="Glob pattern used with --image_dir.")
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH, help="Path to a DeT checkpoint.")
    parser.add_argument("--main_cfg_path", type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument("--output_dir", type=Path, default=Path("demo/output_det"))
    parser.add_argument("--resize", type=int, default=832)
    parser.add_argument("--link_radius", type=float, default=5.0, help="Pixel threshold for linking tracks.")
    parser.add_argument("--det_fine_thr", type=float, default=0.0)
    parser.add_argument("--search_radius", type=float, default=None)
    parser.add_argument("--max_viz_tracks", type=int, default=300)
    parser.add_argument("--viz_height", type=int, default=480)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def collect_images(args) -> List[Path]:
    if args.images is None and args.image_dir is None:
        images = list(DEFAULT_IMAGE_PATHS)
    else:
        images = [resolve_input_path(p) for p in (args.images or [])]
        if args.image_dir is not None:
            image_dir = resolve_input_path(args.image_dir)
            images.extend(sorted(p for p in image_dir.glob(args.pattern) if p.is_file()))

    if len(images) < 3:
        raise ValueError("DeT demo requires at least three ordered images.")

    missing = [str(p) for p in images if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing image files: " + ", ".join(missing))

    return images


def build_config(args) -> dict:
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.JAMMA.MATCH_COARSE.INFERENCE = True
    config.JAMMA.FINE.INFERENCE = True
    config.JAMMA.DET.USE_DET = True
    config.JAMMA.DET.FINE_THR = args.det_fine_thr
    config.JAMMA.DET.SEARCH_RADIUS = (
        args.search_radius if args.search_radius is not None else args.resize * math.sqrt(2)
    )
    config.JAMMA.USE_COMPILE = False
    return lower_config(config)["jamma"]


def load_pair_data(path0: Path, path1: Path, image_id0: int, image_id1: int, args, device):
    image0, scale0, mask0, prepad0 = read_color(path0, args.resize, 8, True)
    image1, scale1, mask1, prepad1 = read_color(path1, args.resize, 8, True)

    mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode="nearest")[0].bool()
    mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode="nearest")[0].bool()

    return {
        "imagec_0": image0.to(device),
        "imagec_1": image1.to(device),
        "mask0": mask0.to(device),
        "mask1": mask1.to(device),
        "scale0": scale0.unsqueeze(0).to(device),
        "scale1": scale1.unsqueeze(0).to(device),
        "prepad_size0": prepad0.unsqueeze(0).to(device),
        "prepad_size1": prepad1.unsqueeze(0).to(device),
        "image_idA": image_id0,
        "image_idB": image_id1,
    }


def tensor_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def keep_prev_data(data: dict) -> dict:
    keys = [
        "image_idB",
        "m_bids",
        "mconf_f",
        "mkpts1_f",
        "mkpts1_subref",
        "mkpts1_f1_fine",
        "mkpts1_f1_window",
    ]
    prev = {}
    for key in keys:
        value = data[key]
        prev[key] = value.detach() if torch.is_tensor(value) else value
    return prev


def run_sequence(model: DeTMatcher, image_paths: List[Path], args, device):
    prev_data: Optional[dict] = None
    pair_results = []

    for idx in range(len(image_paths) - 1):
        data = load_pair_data(image_paths[idx], image_paths[idx + 1], idx, idx + 1, args, device)
        if prev_data is not None:
            data["prev_data"] = prev_data

        logger.info(f"DeT matching: {image_paths[idx]} -> {image_paths[idx + 1]}")
        with torch.no_grad():
            model(data)

        data.pop("prev_data", None)
        pair_results.append(
            {
                "edge": [idx, idx + 1],
                "mkpts0": tensor_to_numpy(data["mkpts0_f_origin"]),
                "mkpts1": tensor_to_numpy(data["mkpts1_f_origin"]),
                "confidence": tensor_to_numpy(data["mconf_f"]),
            }
        )
        prev_data = keep_prev_data(data)

    return pair_results


def build_tracks(pair_results, link_radius: float):
    tracks: List[Dict] = []
    active: Dict[int, int] = {}

    first = pair_results[0]
    for j, (p0, p1) in enumerate(zip(first["mkpts0"], first["mkpts1"])):
        conf = float(first["confidence"][j]) if first["confidence"] is not None else 1.0
        tracks.append(
            {
                "frames": [0, 1],
                "points": [p0.astype(float).tolist(), p1.astype(float).tolist()],
                "confidences": [conf],
            }
        )
        active[len(tracks) - 1] = 1

    for edge_idx, pair in enumerate(pair_results[1:], start=1):
        mk0 = pair["mkpts0"]
        mk1 = pair["mkpts1"]
        conf = pair["confidence"]
        used = set()
        next_active: Dict[int, int] = {}

        for track_id, last_frame in list(active.items()):
            if last_frame != edge_idx or len(mk0) == 0:
                continue

            last_point = np.asarray(tracks[track_id]["points"][-1], dtype=np.float32)
            distances = np.linalg.norm(mk0 - last_point[None, :], axis=1)
            order = np.argsort(distances)
            chosen = None
            for candidate in order:
                candidate = int(candidate)
                if candidate not in used and distances[candidate] <= link_radius:
                    chosen = candidate
                    break

            if chosen is None:
                continue

            used.add(chosen)
            tracks[track_id]["frames"].append(edge_idx + 1)
            tracks[track_id]["points"].append(mk1[chosen].astype(float).tolist())
            tracks[track_id]["confidences"].append(float(conf[chosen]) if conf is not None else 1.0)
            next_active[track_id] = edge_idx + 1

        for j, (p0, p1) in enumerate(zip(mk0, mk1)):
            if j in used:
                continue
            c = float(conf[j]) if conf is not None else 1.0
            tracks.append(
                {
                    "frames": [edge_idx, edge_idx + 1],
                    "points": [p0.astype(float).tolist(), p1.astype(float).tolist()],
                    "confidences": [c],
                }
            )
            next_active[len(tracks) - 1] = edge_idx + 1

        active = next_active

    return tracks


def color_for(index: int):
    r, g, b = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_tracks(image_paths: List[Path], tracks: List[Dict], out_path: Path, max_tracks: int, viz_height: int):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    scaled = []
    scales = []

    for image in images:
        scale = viz_height / image.height
        size = (max(1, int(round(image.width * scale))), viz_height)
        scaled.append(image.resize(size, Image.BICUBIC))
        scales.append((scale, scale))

    offsets = []
    x = 0
    for image in scaled:
        offsets.append(x)
        x += image.width

    canvas = Image.new("RGB", (x, viz_height), (0, 0, 0))
    for image, xoff in zip(scaled, offsets):
        canvas.paste(image, (xoff, 0))

    full_len = len(image_paths)
    ranked = sorted(
        enumerate(tracks),
        key=lambda item: (
            len(item[1]["frames"]) == full_len,
            len(item[1]["frames"]),
            np.mean(item[1]["confidences"]),
        ),
        reverse=True,
    )[:max_tracks]

    draw = ImageDraw.Draw(canvas)
    for draw_idx, (_, track) in enumerate(ranked):
        points = []
        for frame, point in zip(track["frames"], track["points"]):
            sx, sy = scales[frame]
            points.append((offsets[frame] + point[0] * sx, point[1] * sy))

        color = color_for(draw_idx)
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for px, py in points:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    image_paths = collect_images(args)
    if args.ckpt_path != "official":
        args.ckpt_path = str(resolve_input_path(args.ckpt_path))
    args.main_cfg_path = str(resolve_input_path(args.main_cfg_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    model = DeTMatcher(build_config(args), args.ckpt_path).to(device).eval()
    pair_results = run_sequence(model, image_paths, args, device)
    tracks = build_tracks(pair_results, args.link_radius)

    full_tracks = [track for track in tracks if len(track["frames"]) == len(image_paths)]
    payload = {
        "images": [str(path) for path in image_paths],
        "num_pairs": len(pair_results),
        "num_tracks": len(tracks),
        "num_full_tracks": len(full_tracks),
        "pairs": [
            {"edge": pair["edge"], "num_matches": int(len(pair["mkpts0"]))}
            for pair in pair_results
        ],
        "tracks": tracks,
    }

    json_path = args.output_dir / "tracks.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_tracks(image_paths, tracks, args.output_dir / "tracks.png", args.max_viz_tracks, args.viz_height)

    logger.info(f"Saved: {json_path}")
    logger.info(f"Saved: {args.output_dir / 'tracks.png'}")
    logger.info(f"tracks={len(tracks)}, full_tracks={len(full_tracks)}")


if __name__ == "__main__":
    main()
