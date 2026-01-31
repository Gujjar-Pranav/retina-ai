# src/model_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import streamlit as st

import timm


def _unwrap_state_dict(ckpt: Any) -> Dict[str, Any]:
    """
    Supports common checkpoint formats:
    - raw state_dict
    - {"state_dict": ...}
    - {"model": ...}
    - {"model_state_dict": ...}
    - {"model_state": ...}
    """
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "model_state"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # sometimes the dict itself is already a state_dict
        # heuristic: has tensor values
        if any(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    raise ValueError("Checkpoint format not recognized.")


def _looks_like_timm_effnet(sd: Dict[str, Any]) -> bool:
    keys = list(sd.keys())
    return any(k.startswith("conv_stem.") for k in keys) and any(k.startswith("blocks.") for k in keys)


@st.cache_resource
def build_and_load(*, model_path: str, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """
    Builds EfficientNet-B0 as a 2-class classifier [No DR, DR] and loads your checkpoint.
    Returns: (model, backend_string)
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    ckpt = torch.load(p, map_location=device)
    sd = _unwrap_state_dict(ckpt)

    # --- timm EfficientNet-B0 (matches your checkpoint keys: conv_stem/blocks/classifier) ---
    if _looks_like_timm_effnet(sd):
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        model.load_state_dict(sd, strict=True)
        model.to(device)
        model.eval()

        # Optional warmup (safe for Grad-CAM: use no_grad, NOT inference_mode)
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, 224, 224, device=device))

        return model, "timm.efficientnet_b0"

    # --- fallback: try timm but allow partial strictness if user has other formats ---
    # (If you *know* you only use timm checkpoint, you can delete this block)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    try:
        model.load_state_dict(sd, strict=True)
        backend = "timm.efficientnet_b0"
    except Exception:
        # last resort: load non-strict (still keeps app running)
        model.load_state_dict(sd, strict=False)
        backend = "timm.efficientnet_b0 (non-strict)"

    model.to(device)
    model.eval()

    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 224, 224, device=device))

    return model, backend
