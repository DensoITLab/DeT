import time
from typing import Any, Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from eval.records import EvalContext, MethodSpec, PairMatchOutput, PairRequest
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.dataset import read_megadepth_color
from src.utils.profiler import build_profiler


PAIR_MATCHERS: Dict[str, MethodSpec] = {}


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


def build_jamma_config(args, use_det: bool):
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


def get_jamma_model(context: EvalContext, method_name: str, use_det: bool) -> PL_JamMa:
    if method_name not in context.models:
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


def _image_params(args, config) -> Tuple[int, int, bool]:
    resize = args.resize if args.resize is not None else int(config.DATASET.MGDPT_IMG_RESIZE)
    df = args.df if args.df is not None else int(config.DATASET.MGDPT_DF)
    padding = not bool(args.no_padding)
    return resize, df, padding


def _coarse_mask(mask: Optional[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
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
