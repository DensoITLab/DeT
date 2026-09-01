from __future__ import annotations

import argparse
import dataclasses
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
