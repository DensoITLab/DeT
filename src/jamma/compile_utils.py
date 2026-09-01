import os

import torch


def maybe_compile(fn=None, **kwargs):
    enabled = os.environ.get("JAMMA_ENABLE_TORCH_COMPILE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if fn is None:
        return lambda wrapped: maybe_compile(wrapped, **kwargs)
    if not enabled or not hasattr(torch, "compile"):
        return fn
    return torch.compile(fn, **kwargs)
