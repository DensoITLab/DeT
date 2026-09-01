from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class CameraParams:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


@dataclasses.dataclass
class PairRequest:
    img_a: Path
    img_b: Path
    image_id_a: int
    image_id_b: int
    prev_state: Any = None


@dataclasses.dataclass
class PairMatchOutput:
    mkpts0: Any
    mkpts1: Any
    confidence: Optional[Any]
    flops: float
    model_runtime_ms: float
    state: Any = None


@dataclasses.dataclass
class MethodSpec:
    name: str
    runner: Callable[["EvalContext", PairRequest], PairMatchOutput]
    link_mode: str
    use_prev_state: bool


@dataclasses.dataclass
class EvalContext:
    args: argparse.Namespace
    device: Any
    models: Dict[str, Any]
    configs: Dict[str, Any]
    flops_cache: Dict[Tuple[str, bool], float]
    warmup_cache: Dict[Tuple[str, bool], bool]
    timer_events: Dict[str, Tuple[Any, Any]]
    camera_cache: Dict[Tuple[str, str, bool], List[CameraParams]]


PAIR_MATCHERS: Dict[str, MethodSpec] = {}


def add_tracking_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data_cfg_path", type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument("--main_cfg_path", type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument("--ckpt_path", type=str, default="weights/jamma.ckpt")
    parser.add_argument("--dump_dir", type=str, default="dump/eval_tracking")
    parser.add_argument("--profiler_name", type=str, default="inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", type=str, nargs="+", default=["nn-jamma", "det-jamma"])
    parser.add_argument("--bag_size", type=int, default=5)
    parser.add_argument("--topk", type=int, default=20000)
    parser.add_argument("--epi_thr", type=float, default=1e-3)
    parser.add_argument("--nn_link_radius", type=float, default=5.0 * math.sqrt(2.0))
    parser.add_argument("--det_search_radius", type=float, default=832.0 * math.sqrt(2.0))
    parser.add_argument("--det_fine_thr", type=float, default=0.0)
    parser.add_argument("--custom_fine_flex_thr", type=float, default=0.1)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--df", type=int, default=None)
    parser.add_argument("--no_padding", action="store_true")
    parser.add_argument("--thr", type=float, default=None)
    parser.add_argument("--flip_w2c", action="store_true")
    parser.add_argument("--save_json", type=Path, default=None)
    parser.add_argument("--save_summary_json", type=Path, default=None)
    parser.add_argument("--save_errors", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=None)
    return parser


def register_pair_matcher(name: str, link_mode: str, use_prev_state: bool):
    def decorate(fn: Callable[[EvalContext, PairRequest], PairMatchOutput]):
        PAIR_MATCHERS[name] = MethodSpec(
            name=name,
            runner=fn,
            link_mode=link_mode,
            use_prev_state=use_prev_state,
        )
        return fn

    return decorate


def build_jamma_config(args: argparse.Namespace, use_det: bool):
    from src.config.default import get_cfg_defaults

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    config.JAMMA.DET.USE_DET = bool(use_det)
    config.JAMMA.DET.SEARCH_RADIUS = float(args.det_search_radius)
    config.JAMMA.DET.FINE_THR = float(args.det_fine_thr)
    config.JAMMA.USE_COMPILE = False
    if args.thr is not None:
        config.JAMMA.MATCH_COARSE.THR = float(args.thr)
    return config


def get_jamma_model(context: EvalContext, method_name: str, use_det: bool):
    if method_name not in context.models:
        import pytorch_lightning as pl

        from src.lightning.lightning_jamma import PL_JamMa
        from src.utils.profiler import build_profiler

        config = build_jamma_config(context.args, use_det=use_det)
        if not context.configs:
            pl.seed_everything(config.TRAINER.SEED)
        profiler = build_profiler(context.args.profiler_name)
        model = PL_JamMa(
            config,
            pretrained_ckpt=context.args.ckpt_path,
            profiler=profiler,
            dump_dir=context.args.dump_dir,
        )
        context.models[method_name] = model.to(context.device).eval()
        context.configs[method_name] = config
    return context.models[method_name]


def _image_params(args: argparse.Namespace, config) -> Tuple[int, int, bool]:
    resize = args.resize if args.resize is not None else int(config.DATASET.MGDPT_IMG_RESIZE)
    df = args.df if args.df is not None else int(config.DATASET.MGDPT_DF)
    padding = not bool(args.no_padding)
    return resize, df, padding


def _coarse_mask(mask: Optional[Any], image: Any):
    import torch
    import torch.nn.functional as F

    if mask is None:
        mask = torch.ones(image.shape[-2:], dtype=torch.bool)
    return F.interpolate(mask[None, None].float(), scale_factor=0.125, mode="nearest")[0].bool()


def _det_state(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "image_idB",
        "mconf_f",
        "mkpts1_f",
        "mkpts1_subref",
        "mkpts1_f1_fine",
        "mkpts1_f1_window",
        "m_bids",
    )
    return {key: result[key] for key in keys if key in result}


def run_jamma_pair(
    context: EvalContext,
    request: PairRequest,
    method_name: str,
    use_det: bool,
) -> PairMatchOutput:
    import torch

    from src.utils.dataset import read_megadepth_color

    model = get_jamma_model(context, method_name=method_name, use_det=use_det)
    config = context.configs[method_name]
    resize, df, padding = _image_params(context.args, config)

    image0, scale0, mask0, prepad0, *_ = read_megadepth_color(str(request.img_a), resize, df, padding)
    image1, scale1, mask1, prepad1, *_ = read_megadepth_color(str(request.img_b), resize, df, padding)

    data = {
        "imagec_0": image0.to(context.device),
        "imagec_1": image1.to(context.device),
        "mask0": _coarse_mask(mask0, image0).to(context.device),
        "mask1": _coarse_mask(mask1, image1).to(context.device),
        "scale0": scale0.unsqueeze(0).to(context.device),
        "scale1": scale1.unsqueeze(0).to(context.device),
        "prepad_size0": prepad0.unsqueeze(0).to(context.device),
        "prepad_size1": prepad1.unsqueeze(0).to(context.device),
        "custom_fine_flex_thr": float(context.args.custom_fine_flex_thr),
        "image_idA": int(request.image_id_a),
        "image_idB": int(request.image_id_b),
    }
    if request.prev_state is not None:
        data["prev_data"] = request.prev_state

    flops_key = (method_name, request.prev_state is not None)
    if flops_key not in context.flops_cache:
        model._calc_flops_once(data)
        context.flops_cache[flops_key] = float(model._flops_backbone + model._flops_matcher)

    warmup_key = (method_name, request.prev_state is not None)
    with torch.inference_mode():
        if warmup_key not in context.warmup_cache:
            for _ in range(30):
                model.backbone(data)
                model.matcher(data, mode="test")
            context.warmup_cache[warmup_key] = True

        if context.device.type == "cuda":
            if "model" not in context.timer_events:
                context.timer_events["model"] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
            start_event, end_event = context.timer_events["model"]
            torch.cuda.synchronize()
            start_event.record()
            model.backbone(data)
            model.matcher(data, mode="test")
            end_event.record()
            torch.cuda.synchronize()
            model_runtime_ms = float(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            model.backbone(data)
            model.matcher(data, mode="test")
            model_runtime_ms = float((time.perf_counter() - start) * 1000.0)

    result = data
    result.pop("prev_data", None)
    state = _det_state(result) if use_det else None
    return PairMatchOutput(
        mkpts0=result["mkpts0_f_origin"],
        mkpts1=result["mkpts1_f_origin"],
        confidence=result.get("mconf_f"),
        flops=context.flops_cache[flops_key],
        model_runtime_ms=model_runtime_ms,
        state=state,
    )


@register_pair_matcher("nn-jamma", link_mode="nearest", use_prev_state=False)
def run_nn_jamma_pair(context: EvalContext, request: PairRequest) -> PairMatchOutput:
    return run_jamma_pair(context, request, method_name="nn-jamma", use_det=False)


@register_pair_matcher("det-jamma", link_mode="exact", use_prev_state=True)
def run_det_jamma_pair(context: EvalContext, request: PairRequest) -> PairMatchOutput:
    return run_jamma_pair(context, request, method_name="det-jamma", use_det=True)


def _read_cam_from_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

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
    device: Any,
) -> np.ndarray:
    import torch

    from src.utils.metrics import symmetric_epipolar_distance

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


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _finite_float(value: float) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def read_bag_paths(bag_file: Path) -> List[str]:
    return [line.strip() for line in bag_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_bag_files(subset_dir: Path, bag_size: int) -> List[Path]:
    bag_files = sorted(subset_dir.glob(f"{bag_size}bag_*.txt"))
    if not bag_files:
        raise FileNotFoundError(f"No {bag_size}-frame bag files found in {subset_dir}")
    return bag_files


def _to_numpy(value: Any) -> np.ndarray:
    import torch

    if value is None:
        return np.zeros((0,), dtype=np.float64)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_matches(output: PairMatchOutput, topk: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    import torch

    if torch.is_tensor(output.mkpts0) and torch.is_tensor(output.mkpts1):
        mk0_t = output.mkpts0.detach()
        mk1_t = output.mkpts1.detach()
        conf_t = output.confidence.detach() if torch.is_tensor(output.confidence) else None
        n_raw = int(mk0_t.shape[0])

        if conf_t is not None and conf_t.numel() == n_raw and topk > 0 and n_raw > topk:
            keep = torch.topk(conf_t, topk, dim=0, sorted=True).indices
            mk0_t = mk0_t.index_select(0, keep)
            mk1_t = mk1_t.index_select(0, keep)
            conf_t = conf_t.index_select(0, keep)
        elif topk > 0 and n_raw > topk:
            mk0_t = mk0_t[:topk]
            mk1_t = mk1_t[:topk]
            if conf_t is not None:
                conf_t = conf_t[:topk]

        mk0 = mk0_t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 2)
        mk1 = mk1_t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 2)
        conf = None
        if conf_t is not None:
            conf = conf_t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        return mk0, mk1, conf, n_raw

    mk0 = _to_numpy(output.mkpts0).astype(np.float64).reshape(-1, 2)
    mk1 = _to_numpy(output.mkpts1).astype(np.float64).reshape(-1, 2)
    conf = None if output.confidence is None else _to_numpy(output.confidence).astype(np.float64).reshape(-1)
    n_raw = int(mk0.shape[0])

    if conf is not None and conf.size == n_raw and topk > 0 and n_raw > topk:
        keep = np.argpartition(-conf, topk - 1)[:topk]
        keep = keep[np.argsort(-conf[keep])]
        mk0 = mk0[keep]
        mk1 = mk1[keep]
        conf = conf[keep]
    elif topk > 0 and n_raw > topk:
        mk0 = mk0[:topk]
        mk1 = mk1[:topk]
        if conf is not None:
            conf = conf[:topk]

    return mk0, mk1, conf, n_raw


def _point_key(point: np.ndarray) -> Tuple[float, float]:
    return float(point[0]), float(point[1])


def _append_array(points: List[List[float]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _link_exact(
    tracks: Dict[int, Dict[str, Any]],
    next_tid: int,
    pair_index: int,
    mk0: np.ndarray,
    mk1: np.ndarray,
    conf: Optional[np.ndarray],
    prev_point_to_tid: Dict[Tuple[float, float], int],
) -> Tuple[int, Dict[Tuple[float, float], int]]:
    curr_point_to_tid: Dict[Tuple[float, float], int] = {}
    best_by_start: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, float]] = {}

    for match_idx, (pt0, pt1) in enumerate(zip(mk0, mk1)):
        score = float(conf[match_idx]) if conf is not None else 1.0
        key0 = _point_key(pt0)
        if key0 not in best_by_start or score > best_by_start[key0][2]:
            best_by_start[key0] = (pt0, pt1, score)

    for key0, (pt0, pt1, score) in best_by_start.items():
        key1 = _point_key(pt1)
        if key0 in prev_point_to_tid:
            tid = prev_point_to_tid[key0]
            track = tracks[tid]
            track["points"].append([float(pt1[0]), float(pt1[1])])
            track["end_id"] = pair_index + 1
            track["scores"].append(score)
        else:
            tid = next_tid
            next_tid += 1
            tracks[tid] = {
                "start_id": pair_index,
                "end_id": pair_index + 1,
                "points": [[float(pt0[0]), float(pt0[1])], [float(pt1[0]), float(pt1[1])]],
                "scores": [score],
            }
        curr_point_to_tid[key1] = tid

    return next_tid, curr_point_to_tid


def _link_nearest(
    tracks: Dict[int, Dict[str, Any]],
    next_tid: int,
    pair_index: int,
    mk0: np.ndarray,
    mk1: np.ndarray,
    conf: Optional[np.ndarray],
    prev_points: Optional[np.ndarray],
    prev_tids: Optional[np.ndarray],
    radius: float,
) -> Tuple[int, np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    curr_points: List[List[float]] = []
    curr_tids: List[int] = []
    max_d2 = float(radius * radius)

    if prev_points is not None and prev_points.shape[0] > 0:
        used = np.zeros(mk0.shape[0], dtype=bool)
        if mk0.shape[0] > 0 and cKDTree is not None:
            distances, nearest = cKDTree(mk0).query(prev_points, k=1, distance_upper_bound=radius)
            candidate_prev = np.flatnonzero(nearest < mk0.shape[0])
        else:
            distances = None
            nearest = None
            candidate_prev = range(prev_points.shape[0])

        for prev_idx in candidate_prev:
            if nearest is None:
                if mk0.shape[0] == 0:
                    break
                d2 = np.sum((mk0 - prev_points[prev_idx][None, :]) ** 2, axis=1)
                match_idx = int(np.argmin(d2))
                if float(d2[match_idx]) > max_d2:
                    continue
            else:
                match_idx = int(nearest[prev_idx])
                if float(distances[prev_idx] * distances[prev_idx]) > max_d2:
                    continue

            if used[match_idx]:
                continue

            used[match_idx] = True
            tid = int(prev_tids[prev_idx])
            pt1 = mk1[match_idx]
            score = float(conf[match_idx]) if conf is not None else 1.0
            tracks[tid]["points"].append([float(pt1[0]), float(pt1[1])])
            tracks[tid]["end_id"] = pair_index + 1
            tracks[tid]["scores"].append(score)
            curr_points.append([float(pt1[0]), float(pt1[1])])
            curr_tids.append(tid)
    else:
        used = np.zeros(mk0.shape[0], dtype=bool)

    for match_idx, (pt0, pt1) in enumerate(zip(mk0, mk1)):
        if used[match_idx]:
            continue
        score = float(conf[match_idx]) if conf is not None else 1.0
        tid = next_tid
        next_tid += 1
        tracks[tid] = {
            "start_id": pair_index,
            "end_id": pair_index + 1,
            "points": [[float(pt0[0]), float(pt0[1])], [float(pt1[0]), float(pt1[1])]],
            "scores": [score],
        }
        curr_points.append([float(pt1[0]), float(pt1[1])])
        curr_tids.append(tid)

    return next_tid, _append_array(curr_points), np.asarray(curr_tids, dtype=np.int64)


def evaluate_bag(
    bag_file: Path,
    method: MethodSpec,
    context: EvalContext,
    dataset_root: Path,
    calib_dir: Path,
) -> Dict[str, Any]:
    args = context.args
    rel_paths = read_bag_paths(bag_file)
    if len(rel_paths) != args.bag_size:
        raise ValueError(f"{bag_file}: expected {args.bag_size} paths, got {len(rel_paths)}")

    img_paths = [dataset_root / rel_path for rel_path in rel_paths]
    camera_key = (str(calib_dir), str(bag_file), bool(args.flip_w2c))
    if camera_key not in context.camera_cache:
        context.camera_cache[camera_key] = [
            load_cam_from_dir(calib_dir, img_path, args.flip_w2c) for img_path in img_paths
        ]
    cameras = context.camera_cache[camera_key]

    tracks: Dict[int, Dict[str, Any]] = {}
    next_tid = 0
    prev_state = None
    prev_point_to_tid: Dict[Tuple[float, float], int] = {}
    prev_points: Optional[np.ndarray] = None
    prev_tids: Optional[np.ndarray] = None
    pair_summaries: List[Dict[str, Any]] = []
    flops_total = 0.0
    model_runtime_ms_total = 0.0

    for pair_index in range(args.bag_size - 1):
        request = PairRequest(
            img_a=img_paths[pair_index],
            img_b=img_paths[pair_index + 1],
            image_id_a=pair_index,
            image_id_b=pair_index + 1,
            prev_state=prev_state if method.use_prev_state else None,
        )
        output = method.runner(context, request)
        if method.use_prev_state:
            prev_state = output.state

        mk0, mk1, conf, raw_matches = _prepare_matches(output, args.topk)
        flops_total += float(output.flops)
        model_runtime_ms_total += float(output.model_runtime_ms)
        pair_summaries.append(
            {
                "pair_index": pair_index,
                "image_a": rel_paths[pair_index],
                "image_b": rel_paths[pair_index + 1],
                "raw_matches": raw_matches,
                "used_matches": int(mk0.shape[0]),
                "flops": float(output.flops),
                "model_runtime_ms": float(output.model_runtime_ms),
            }
        )

        if method.link_mode == "exact":
            next_tid, prev_point_to_tid = _link_exact(
                tracks=tracks,
                next_tid=next_tid,
                pair_index=pair_index,
                mk0=mk0,
                mk1=mk1,
                conf=conf,
                prev_point_to_tid=prev_point_to_tid,
            )
        elif method.link_mode == "nearest":
            next_tid, prev_points, prev_tids = _link_nearest(
                tracks=tracks,
                next_tid=next_tid,
                pair_index=pair_index,
                mk0=mk0,
                mk1=mk1,
                conf=conf,
                prev_points=prev_points,
                prev_tids=prev_tids,
                radius=args.nn_link_radius,
            )
        else:
            raise ValueError(f"Unknown link mode: {method.link_mode}")

    tracks_start_at_0 = 0
    endpoints_0: List[List[float]] = []
    endpoints_n: List[List[float]] = []
    for track in tracks.values():
        points = track.get("points", [])
        if track.get("start_id") == 0:
            tracks_start_at_0 += 1
            if track.get("end_id") == args.bag_size - 1 and len(points) == args.bag_size:
                endpoints_0.append(points[0])
                endpoints_n.append(points[-1])

    endpoints_0_np = _append_array(endpoints_0)
    endpoints_n_np = _append_array(endpoints_n)
    epi_errors = symmetric_epi_errors(endpoints_0_np, endpoints_n_np, cameras[0], cameras[-1], context.device)
    correct = int((epi_errors < args.epi_thr).sum())
    evaluated = int(epi_errors.size)
    precision = float(correct / evaluated) if evaluated else 0.0
    num_pairs = args.bag_size - 1

    result = {
        "bag_file": str(bag_file),
        "images": rel_paths,
        "num_pairs": int(num_pairs),
        "pair_summaries": pair_summaries,
        "tracks_start_at_0": int(tracks_start_at_0),
        "full_tracks_0_to_N": int(evaluated),
        "correct_tracks_0_to_N": int(correct),
        "precision_0_to_N": precision,
        "mean_epi_error_0_to_N": _finite_float(np.mean(epi_errors)) if evaluated else None,
        "median_epi_error_0_to_N": _finite_float(np.median(epi_errors)) if evaluated else None,
        "epi_threshold": float(args.epi_thr),
        "flops_total": float(flops_total),
        "model_runtime_ms_total": float(model_runtime_ms_total),
        "avg_flops_per_pair_GMac": float((flops_total / num_pairs) / 1e9) if num_pairs else 0.0,
        "avg_model_runtime_per_pair_ms": float(model_runtime_ms_total / num_pairs) if num_pairs else 0.0,
    }
    if args.save_errors:
        result["epi_errors_0_to_N"] = epi_errors.tolist()
    return result


def aggregate_results(per_bag: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_bag:
        return {
            "bags_total": 0,
            "avg_correct_tracks_0_to_N": 0.0,
            "avg_flops_per_pair_GMac": 0.0,
            "avg_model_runtime_per_pair_ms": 0.0,
        }

    correct = np.asarray([r["correct_tracks_0_to_N"] for r in per_bag], dtype=np.float64)
    full_tracks = np.asarray([r["full_tracks_0_to_N"] for r in per_bag], dtype=np.float64)
    start_tracks = np.asarray([r["tracks_start_at_0"] for r in per_bag], dtype=np.float64)
    total_pairs = int(sum(r["num_pairs"] for r in per_bag))
    total_flops = float(sum(r["flops_total"] for r in per_bag))
    total_model_runtime = float(sum(r["model_runtime_ms_total"] for r in per_bag))
    total_correct = int(correct.sum())
    total_evaluated = int(full_tracks.sum())
    avg_flops_gmac = float((total_flops / total_pairs) / 1e9) if total_pairs else 0.0
    avg_model_runtime_ms = float(total_model_runtime / total_pairs) if total_pairs else 0.0

    return {
        "bags_total": int(len(per_bag)),
        "bags_with_full_tracks": int(np.sum(full_tracks > 0)),
        "total_pairs": total_pairs,
        "total_correct_tracks_0_to_N": total_correct,
        "total_evaluated_tracks_0_to_N": total_evaluated,
        "global_precision_0_to_N": float(total_correct / total_evaluated) if total_evaluated else 0.0,
        "avg_tracks_start_at_0": float(start_tracks.mean()),
        "avg_full_tracks_0_to_N": float(full_tracks.mean()),
        "avg_correct_tracks_0_to_N": float(correct.mean()),
        "std_correct_tracks_0_to_N": float(correct.std(ddof=0)),
        "avg_flops_per_pair_GMac": avg_flops_gmac,
        "avg_model_runtime_per_pair_ms": avg_model_runtime_ms,
        "total_flops_GMac": float(total_flops / 1e9),
        "total_model_runtime_ms": total_model_runtime,
    }


def run_tracking_evaluation(
    args: argparse.Namespace,
    dataset_root: Path,
    subset_dir: Path,
    calib_dir: Path,
) -> Dict[str, Any]:
    import torch

    if args.dataset_name is None:
        raise ValueError("--dataset_name must be set")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    selected_methods = list(dict.fromkeys(args.methods))
    unknown = [name for name in selected_methods if name not in PAIR_MATCHERS]
    if unknown:
        available = ", ".join(sorted(PAIR_MATCHERS.keys()))
        raise ValueError(f"Unknown method(s): {unknown}. Available methods: {available}")

    context = EvalContext(
        args=args,
        device=device,
        models={},
        configs={},
        flops_cache={},
        warmup_cache={},
        timer_events={},
        camera_cache={},
    )
    bag_files = find_bag_files(subset_dir, args.bag_size)
    results: Dict[str, Dict[str, Any]] = {}

    for method_name in selected_methods:
        method = PAIR_MATCHERS[method_name]
        per_bag: List[Dict[str, Any]] = []
        for bag_idx, bag_file in enumerate(bag_files):
            print(f"[{method_name}] {bag_idx + 1}/{len(bag_files)} {bag_file.name}")
            per_bag.append(
                evaluate_bag(
                    bag_file=bag_file,
                    method=method,
                    context=context,
                    dataset_root=dataset_root,
                    calib_dir=calib_dir,
                )
            )
        results[method_name] = {
            "summary": aggregate_results(per_bag),
            "per_bag": per_bag,
        }

    output = {
        "dataset_name": args.dataset_name,
        "scene_name": getattr(args, "scene_name", None),
        "bag_size": int(args.bag_size),
        "epi_threshold": float(args.epi_thr),
        "subset_dir": str(subset_dir),
        "dataset_root": str(dataset_root),
        "calib_dir": str(calib_dir),
        "methods": results,
    }

    summary = {
        "dataset_name": output["dataset_name"],
        "scene_name": output["scene_name"],
        "bag_size": output["bag_size"],
        "epi_threshold": output["epi_threshold"],
        "methods": {name: data["summary"] for name, data in results.items()},
    }

    print("\nmethod\tcorrect_tracks\tflops_GMac\tmodel_time_ms\tprecision")
    for method_name, data in results.items():
        s = data["summary"]
        print(
            f"{method_name}\t"
            f"{s['avg_correct_tracks_0_to_N']:.2f}\t"
            f"{s['avg_flops_per_pair_GMac']:.4f}\t"
            f"{s['avg_model_runtime_per_pair_ms']:.4f}\t"
            f"{s['global_precision_0_to_N']:.4f}"
        )

    if args.save_json is not None:
        write_json(args.save_json, output)
        print(f"Wrote {args.save_json}")
    if args.save_summary_json is not None:
        write_json(args.save_summary_json, summary)
        print(f"Wrote {args.save_summary_json}")

    return output
