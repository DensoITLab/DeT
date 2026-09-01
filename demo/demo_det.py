#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.default import get_cfg_defaults
from src.jamma.backbone import CovNextV2_nano
from src.jamma.jamma import JamMa

DEFAULT_IMAGE_PATHS = [
    PROJECT_ROOT / "assets" / "phototourism_sample_images" / "piazza_san_marco_06795901_3725050516.jpg",
    PROJECT_ROOT / "assets" / "phototourism_sample_images" / "piazza_san_marco_15148634_5228701572.jpg",
    PROJECT_ROOT / "assets" / "phototourism_sample_images" / "piazza_san_marco_18627786_5929294590.jpg",
]
DEFAULT_CKPT_PATH = "weights/jamma.ckpt"
TRACK_COLOR = (18, 108, 98)
NN_FOCUS_COLOR = (0, 176, 82)
DET_FOCUS_COLOR = (226, 63, 48)
POINT_OUTLINE_COLOR = (3, 24, 24)
ZOOM_BORDER_COLOR = (42, 99, 176)


class JamMaDemoMatcher(torch.nn.Module):
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
        help="Ordered image paths. Defaults to the first three Piazza San Marco images in assets/phototourism_sample_images.",
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
    parser.add_argument("--max_viz_tracks", type=int, default=90)
    parser.add_argument("--viz_height", type=int, default=720)
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--line_alpha", type=int, default=120)
    parser.add_argument("--point_radius", type=int, default=3)
    parser.add_argument("--point_alpha", type=int, default=210)
    parser.add_argument("--zoom_frame", type=int, default=2)
    parser.add_argument("--zoom_window_size", type=int, default=320)
    parser.add_argument("--zoom_crop_size", type=float, default=42.0)
    parser.add_argument("--zoom_seed", type=int, default=0, help="Random seed for reproducible zoom selection.")
    parser.add_argument("--track_spacing", type=float, default=10.0)
    parser.add_argument("--label_font_size", type=int, default=28)
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


def build_config(args, use_det: bool) -> dict:
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.JAMMA.MATCH_COARSE.INFERENCE = True
    config.JAMMA.FINE.INFERENCE = True
    config.JAMMA.DET.USE_DET = use_det
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


def run_sequence(model: JamMaDemoMatcher, image_paths: List[Path], args, device, use_det: bool, label: str):
    prev_data: Optional[dict] = None
    pair_results = []

    for idx in range(len(image_paths) - 1):
        data = load_pair_data(image_paths[idx], image_paths[idx + 1], idx, idx + 1, args, device)
        if use_det and prev_data is not None:
            data["prev_data"] = prev_data

        logger.info(f"{label}: {image_paths[idx]} -> {image_paths[idx + 1]}")
        with torch.inference_mode():
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
        if use_det:
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point_for_frame(track: Dict, frame_index: int) -> Optional[np.ndarray]:
    for frame, point in zip(track["frames"], track["points"]):
        if frame == frame_index:
            return np.asarray(point, dtype=np.float32)
    return None


def draw_point(
    draw: ImageDraw.ImageDraw,
    point: Tuple[float, float],
    color: Tuple[int, int, int],
    radius: int,
    alpha: int,
) -> None:
    px, py = point
    radius = max(1, int(radius))
    fill_alpha = max(0, min(255, int(alpha)))
    outline_alpha = max(0, min(255, int(alpha) + 30))
    draw.ellipse(
        (px - radius, py - radius, px + radius, py + radius),
        fill=(*color, fill_alpha),
        outline=(*POINT_OUTLINE_COLOR, outline_alpha),
    )


def starts_from_first_pair(track: Dict) -> bool:
    return len(track["frames"]) >= 2 and track["frames"][0] == 0 and track["frames"][1] == 1


def select_common_start_track_ids(
    nn_tracks: List[Dict],
    det_tracks: List[Dict],
    common_start_count: int,
    max_tracks: int,
    min_spacing: float,
) -> List[int]:
    candidates = []
    count = min(common_start_count, len(nn_tracks), len(det_tracks))
    for track_id in range(count):
        nn_track = nn_tracks[track_id]
        det_track = det_tracks[track_id]
        if not starts_from_first_pair(nn_track) or not starts_from_first_pair(det_track):
            continue
        nn_conf = float(np.mean(nn_track["confidences"])) if nn_track["confidences"] else 0.0
        det_conf = float(np.mean(det_track["confidences"])) if det_track["confidences"] else 0.0
        candidates.append((track_id, len(nn_track["frames"]) + len(det_track["frames"]), nn_conf + det_conf))

    ranked = sorted(
        candidates,
        key=lambda item: (
            item[1],
            item[2],
        ),
        reverse=True,
    )

    selected: List[int] = []
    anchors: List[np.ndarray] = []
    for track_id, _, _ in ranked:
        point = np.asarray(nn_tracks[track_id]["points"][0], dtype=np.float32)
        if any(np.linalg.norm(point - prev_point) < min_spacing for prev_point in anchors):
            continue
        selected.append(track_id)
        anchors.append(point)
        if len(selected) >= max_tracks:
            break

    return selected


def select_zoom_region(
    nn_tracks: List[Dict],
    det_tracks: List[Dict],
    common_start_count: int,
    frame_index: int,
    base_crop_size: float,
    seed: Optional[int],
) -> Optional[Dict]:
    candidates = []
    count = min(common_start_count, len(nn_tracks), len(det_tracks))

    for track_id in range(count):
        nn_track = nn_tracks[track_id]
        det_track = det_tracks[track_id]
        if not starts_from_first_pair(nn_track) or not starts_from_first_pair(det_track):
            continue

        nn_point = point_for_frame(nn_track, frame_index)
        det_point = point_for_frame(det_track, frame_index)
        if nn_point is None or det_point is None:
            continue

        gap = float(np.linalg.norm(nn_point - det_point))
        if not np.isfinite(gap):
            continue

        candidates.append((track_id, gap, nn_point, det_point))

    if not candidates:
        return None

    rng = np.random.default_rng(seed)
    track_id, gap, nn_point, det_point = candidates[int(rng.integers(len(candidates)))]
    center = (nn_point + det_point) * 0.5
    crop_size = max(float(base_crop_size), gap + float(base_crop_size) * 0.75)
    return {
        "frame": frame_index,
        "track_id": track_id,
        "center": center.tolist(),
        "crop_size": crop_size,
        "gap": gap,
        "nn_point": nn_point.tolist(),
        "det_point": det_point.tolist(),
    }


def crop_box_for(image: Image.Image, center: List[float], crop_size: float) -> Tuple[int, int, int, int]:
    size = int(round(max(2.0, min(float(crop_size), float(image.width), float(image.height)))))
    left = int(round(clamp(float(center[0]) - size * 0.5, 0, image.width - size)))
    top = int(round(clamp(float(center[1]) - size * 0.5, 0, image.height - size)))
    return left, top, left + size, top + size


def load_label_font(size: int):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_row(
    image_paths: List[Path],
    tracks: List[Dict],
    label: str,
    label_fill: Tuple[int, int, int],
    selected_track_ids: List[int],
    max_tracks: int,
    viz_height: int,
    line_width: int,
    line_alpha: int,
    point_radius: int,
    point_alpha: int,
    zoom_region: Optional[Dict],
    focus_color: Tuple[int, int, int],
    label_font_size: int,
) -> Image.Image:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    scaled = []
    scales = []

    for image in images:
        scale = viz_height / image.height
        size = (max(1, int(round(image.width * scale))), viz_height)
        scaled.append(image.resize(size, Image.Resampling.LANCZOS))
        scales.append((scale, scale))

    offsets = []
    x = 0
    for image in scaled:
        offsets.append(x)
        x += image.width

    canvas = Image.new("RGBA", (x, viz_height), (0, 0, 0, 255))
    for image, xoff in zip(scaled, offsets):
        canvas.paste(image.convert("RGBA"), (xoff, 0))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    visible_track_ids = selected_track_ids[:max_tracks]
    focus_track_id = int(zoom_region["track_id"]) if zoom_region is not None else None
    for track_id in visible_track_ids:
        if track_id >= len(tracks):
            continue
        track = tracks[track_id]
        points = []
        for frame, point in zip(track["frames"], track["points"]):
            sx, sy = scales[frame]
            points.append((offsets[frame] + point[0] * sx, point[1] * sy))

        if len(points) > 1:
            draw.line(points, fill=(*TRACK_COLOR, max(0, min(255, line_alpha))), width=max(1, line_width))
        is_focus_track = focus_track_id is not None and track_id == focus_track_id
        point_color = focus_color if is_focus_track else TRACK_COLOR
        point_draw_alpha = 255 if is_focus_track else point_alpha
        point_draw_radius = point_radius + 1 if is_focus_track else point_radius
        for point in points:
            draw_point(draw, point, point_color, point_draw_radius, point_draw_alpha)

    canvas = Image.alpha_composite(canvas, overlay)
    label_draw = ImageDraw.Draw(canvas)
    font = load_label_font(max(12, label_font_size))
    bbox = label_draw.textbbox((0, 0), label, font=font)
    pad_x = max(6, int(label_font_size * 0.25))
    pad_y = max(3, int(label_font_size * 0.15))
    rect = (0, 0, bbox[2] - bbox[0] + pad_x * 2, bbox[3] - bbox[1] + pad_y * 2)
    label_draw.rectangle(rect, fill=(*label_fill, 245))
    label_draw.text((pad_x, pad_y), label, fill=(0, 0, 0, 255), font=font)
    return canvas.convert("RGB")


def get_image_layout(image_paths: List[Path], viz_height: int):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    offsets = []
    scales = []
    width = 0
    for image in images:
        scale = viz_height / image.height
        offsets.append(width)
        scales.append((scale, scale))
        width += max(1, int(round(image.width * scale)))
    return images, offsets, scales, width


def draw_shared_zoom(
    canvas: Image.Image,
    image_paths: List[Path],
    zoom_region: Optional[Dict],
    row_height: int,
    row_gap: int,
    zoom_window_size: int,
    line_width: int,
    point_radius: int,
) -> Image.Image:
    if zoom_region is None or zoom_window_size <= 0:
        return canvas

    frame_index = int(zoom_region["frame"])
    images, offsets, scales, row_width = get_image_layout(image_paths, row_height)
    if frame_index < 0 or frame_index >= len(images):
        return canvas

    zoom_size = min(row_height, max(64, int(zoom_window_size)))
    zoom_gap = max(8, int(row_height * 0.025))
    zoom_x = row_width + zoom_gap
    zoom_y = int(round((canvas.height - zoom_size) * 0.5))
    zoom_box = crop_box_for(images[frame_index], zoom_region["center"], zoom_region["crop_size"])
    left, top, right, bottom = zoom_box
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)

    canvas_rgba = canvas.convert("RGBA")
    zoom_source = images[frame_index].crop(zoom_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    canvas_rgba.paste(zoom_source.convert("RGBA"), (zoom_x, zoom_y))

    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    border = (*ZOOM_BORDER_COLOR, 230)
    connector = (*ZOOM_BORDER_COLOR, 170)
    line_width_zoom = max(2, line_width + 1)
    sx, sy = scales[frame_index]
    dst_box = (zoom_x, zoom_y, zoom_x + zoom_size, zoom_y + zoom_size)
    draw.rectangle(dst_box, outline=border, width=line_width_zoom)

    for row_index, row_y in enumerate((0, row_height + row_gap)):
        src_box = (
            offsets[frame_index] + left * sx,
            row_y + top * sy,
            offsets[frame_index] + right * sx,
            row_y + bottom * sy,
        )
        target_y = dst_box[1] if row_index == 0 else dst_box[3]
        draw.rectangle(src_box, outline=border, width=line_width_zoom)
        draw.line((src_box[2], src_box[1], dst_box[0], target_y), fill=connector, width=line_width_zoom)
        draw.line((src_box[2], src_box[3], dst_box[0], target_y), fill=connector, width=line_width_zoom)

    def to_zoom(point: List[float]) -> Tuple[float, float]:
        return (
            zoom_x + (float(point[0]) - left) * zoom_size / crop_width,
            zoom_y + (float(point[1]) - top) * zoom_size / crop_height,
        )

    focus_radius = max(point_radius * 2, 6)
    draw_point(draw, to_zoom(zoom_region["nn_point"]), NN_FOCUS_COLOR, focus_radius, 255)
    draw_point(draw, to_zoom(zoom_region["det_point"]), DET_FOCUS_COLOR, focus_radius, 255)

    return Image.alpha_composite(canvas_rgba, overlay).convert("RGB")


def draw_comparison(
    image_paths: List[Path],
    nn_tracks: List[Dict],
    det_tracks: List[Dict],
    common_start_count: int,
    out_path: Path,
    max_tracks: int,
    viz_height: int,
    line_width: int,
    line_alpha: int,
    point_radius: int,
    point_alpha: int,
    zoom_frame: int,
    zoom_window_size: int,
    zoom_crop_size: float,
    zoom_seed: Optional[int],
    track_spacing: float,
    label_font_size: int,
) -> Tuple[List[int], Optional[Dict]]:
    selected_track_ids = select_common_start_track_ids(
        nn_tracks,
        det_tracks,
        common_start_count,
        max_tracks,
        track_spacing,
    )
    zoom_region = select_zoom_region(
        nn_tracks,
        det_tracks,
        common_start_count,
        zoom_frame,
        zoom_crop_size,
        zoom_seed,
    )
    if zoom_region is not None and int(zoom_region["track_id"]) not in selected_track_ids:
        selected_track_ids = [int(zoom_region["track_id"])] + selected_track_ids
    selected_track_ids = selected_track_ids[:max_tracks]
    nn_row = build_row(
        image_paths, nn_tracks, "NN-JamMa", (255, 255, 255), selected_track_ids,
        max_tracks, viz_height, line_width, line_alpha, point_radius, point_alpha,
        zoom_region, NN_FOCUS_COLOR, label_font_size
    )
    det_row = build_row(
        image_paths, det_tracks, "DeT-JamMa", (255, 205, 0), selected_track_ids,
        max_tracks, viz_height, line_width, line_alpha, point_radius, point_alpha,
        zoom_region, DET_FOCUS_COLOR, label_font_size
    )

    gap = max(10, int(viz_height * 0.08))
    row_width = max(nn_row.width, det_row.width)
    zoom_extra_width = 0
    if zoom_region is not None and zoom_window_size > 0:
        zoom_extra_width = max(8, int(viz_height * 0.025)) + min(viz_height, max(64, int(zoom_window_size)))
    width = row_width + zoom_extra_width
    canvas = Image.new("RGB", (width, nn_row.height + gap + det_row.height), (255, 255, 255))
    canvas.paste(nn_row, (0, 0))
    canvas.paste(det_row, (0, nn_row.height + gap))
    canvas = draw_shared_zoom(
        canvas,
        image_paths,
        zoom_region,
        nn_row.height,
        gap,
        zoom_window_size,
        line_width,
        point_radius,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return selected_track_ids, zoom_region


def main():
    args = parse_args()
    image_paths = collect_images(args)
    if args.ckpt_path != "official":
        args.ckpt_path = str(resolve_input_path(args.ckpt_path))
    args.main_cfg_path = str(resolve_input_path(args.main_cfg_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(args.device)
    device = requested_device
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    logger.info(f"device={device}")

    nn_model = JamMaDemoMatcher(build_config(args, use_det=False), args.ckpt_path).to(device).eval()
    nn_pair_results = run_sequence(nn_model, image_paths, args, device, use_det=False, label="NN-JamMa")
    nn_tracks = build_tracks(nn_pair_results, args.link_radius)
    del nn_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    det_model = JamMaDemoMatcher(build_config(args, use_det=True), args.ckpt_path).to(device).eval()
    det_pair_results = run_sequence(det_model, image_paths, args, device, use_det=True, label="DeT-JamMa")
    det_tracks = build_tracks(det_pair_results, args.link_radius)

    common_start_count = 0
    if nn_pair_results and det_pair_results:
        common_start_count = min(len(nn_pair_results[0]["mkpts0"]), len(det_pair_results[0]["mkpts0"]))

    comparison_path = args.output_dir / "comparison.png"
    selected_track_ids, zoom_region = draw_comparison(
        image_paths,
        nn_tracks,
        det_tracks,
        common_start_count,
        comparison_path,
        args.max_viz_tracks,
        args.viz_height,
        args.line_width,
        args.line_alpha,
        args.point_radius,
        args.point_alpha,
        args.zoom_frame,
        args.zoom_window_size,
        args.zoom_crop_size,
        args.zoom_seed,
        args.track_spacing,
        args.label_font_size,
    )

    payload = {
        "images": [str(path) for path in image_paths],
        "comparison": {
            "common_start_count": common_start_count,
            "selected_track_ids": selected_track_ids,
            "zoom_region": zoom_region,
        },
        "methods": {
            "nn-jamma": {
                "num_pairs": len(nn_pair_results),
                "num_tracks": len(nn_tracks),
                "num_full_tracks": sum(len(track["frames"]) == len(image_paths) for track in nn_tracks),
                "pairs": [
                    {"edge": pair["edge"], "num_matches": int(len(pair["mkpts0"]))}
                    for pair in nn_pair_results
                ],
                "tracks": nn_tracks,
            },
            "det-jamma": {
                "num_pairs": len(det_pair_results),
                "num_tracks": len(det_tracks),
                "num_full_tracks": sum(len(track["frames"]) == len(image_paths) for track in det_tracks),
                "pairs": [
                    {"edge": pair["edge"], "num_matches": int(len(pair["mkpts0"]))}
                    for pair in det_pair_results
                ],
                "tracks": det_tracks,
            },
        },
    }

    json_path = args.output_dir / "tracks.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(f"Saved: {json_path}")
    logger.info(f"Saved: {comparison_path}")
    logger.info(
        f"tracks: nn-jamma={len(nn_tracks)}, det-jamma={len(det_tracks)}, "
        f"common_tracks={len(selected_track_ids)}"
    )


if __name__ == "__main__":
    main()
