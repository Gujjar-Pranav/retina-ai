# src/screening_core.py
from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F


IMG_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _as_bool(v: Any) -> bool:
    """
    Robust boolean parser for Excel/pandas values.
    Treats only true-like values as True.
    - True, 1, "yes", "true", "1", "y" => True
    - False, 0, "no", "false", "", None, NaN => False
    """
    try:
        if v is None:
            return False
        # pandas NaN
        if isinstance(v, float) and pd.isna(v):
            return False
        if pd.isna(v):
            return False
    except Exception:
        pass

    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v) == 1
    if isinstance(v, (float, np.floating)):
        # treat 1.0 as True, everything else False
        try:
            return float(v) == 1.0
        except Exception:
            return False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"yes", "true", "1", "y"}:
            return True
        if s in {"no", "false", "0", "n", ""}:
            return False
        return False

    return False


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(chw)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0)  # [1,3,224,224]


def compute_quality_metrics(img: Image.Image) -> Dict[str, float]:
    rgb = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)

    g = gray
    lap = (
        k[1, 1] * g
        + k[0, 1] * np.roll(g, -1, axis=0)
        + k[2, 1] * np.roll(g, 1, axis=0)
        + k[1, 0] * np.roll(g, -1, axis=1)
        + k[1, 2] * np.roll(g, 1, axis=1)
    )
    sharpness_proxy = float(np.var(lap))

    return {
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "sharpness_proxy": sharpness_proxy,
    }


@torch.no_grad()
def predict_2class(model: torch.nn.Module, device: torch.device, img: Image.Image) -> Dict[str, Any]:
    model.eval()
    x = pil_to_tensor(img).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    p_no_dr = float(probs[0].item())
    p_dr = float(probs[1].item())
    pred_idx = int(torch.argmax(probs).item())
    pred = "DR" if pred_idx == 1 else "No DR"
    confidence = float(probs[pred_idx].item())
    return {"p_dr": p_dr, "p_no_dr": p_no_dr, "pred": pred, "confidence": confidence}


def risk_stratification(p_dr: float) -> Tuple[str, str]:
    if p_dr < 0.20:
        return "Low", "Annual"
    if p_dr < 0.50:
        return "Mild", "6–12 months"
    if p_dr < 0.80:
        return "Moderate", "3–6 months"
    return "High", "1–4 weeks"


def derive_risk_factors(patient_row: Dict[str, Any]) -> List[str]:
    r: List[str] = []

    # Diabetes duration
    try:
        yrs = patient_row.get("diabetes_years", patient_row.get("diabetes_duration_years", None))
        if pd.notna(yrs) and str(yrs).strip() != "":
            if float(yrs) >= 10:
                r.append("Diabetes duration ≥ 10 years")
    except Exception:
        pass

    # Hypertension (FIXED)
    try:
        if _as_bool(patient_row.get("hypertension")):
            r.append("Hypertension")
    except Exception:
        pass

    # HbA1c (FIXED fallback)
    try:
        a1c = patient_row.get("last_hba1c", patient_row.get("hba1c", None))
        if pd.notna(a1c) and str(a1c).strip() != "":
            if float(a1c) >= 7.5:
                r.append("Elevated HbA1c (≥ 7.5)")
    except Exception:
        pass

    return r if r else ["No major risk factors recorded"]


def build_recommendation(pred: str, risk_level: str, followup: str, quality_flagged: bool, confidence_flagged: bool) -> str:
    if quality_flagged:
        return (
            "Image quality is suboptimal (RETAKE recommended). "
            "Repeat fundus capture. If repeat remains suboptimal, route to clinician review."
        )
    if confidence_flagged:
        return (
            "Model confidence is below the configured threshold (REVIEW recommended). "
            "Route to clinician for confirmation and management decisions."
        )

    if pred == "No DR":
        return (
            "No diabetic retinopathy detected by AI screening. "
            f"Recommend routine follow-up: {followup}. "
            "If symptoms occur (vision changes, floaters), seek prompt evaluation."
        )

    if risk_level in ("High", "Moderate"):
        return (
            "Diabetic retinopathy suspected. "
            f"Recommend ophthalmology referral and follow-up within {followup}. "
            "Optimize glycemic control (HbA1c), blood pressure, and lipid management per clinical guidance."
        )

    return (
        "Diabetic retinopathy suspected (lower risk band). "
        f"Recommend follow-up within {followup} and consider ophthalmology review. "
        "Reinforce diabetes control and routine eye screening."
    )


class GradCAM:
    """
    Fixed: always upsamples CAM to 224x224 (no more 7x7 -> 224 broadcast crash).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_inp, grad_out):
        self.gradients = grad_out[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, x: torch.Tensor, target_index: int = 1) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                 # [1,2]
        score = logits[:, target_index].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed. Target layer may be wrong.")

        w = self.gradients.mean(dim=(2, 3), keepdim=True)           # [1,C,1,1]
        cam = (w * self.activations).sum(dim=1, keepdim=True)       # [1,1,h,w]
        cam = F.relu(cam)

        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)

        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return cam[0, 0].detach().cpu().numpy().astype(np.float32)


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found for Grad-CAM.")
    return last


def overlay_cam_on_image(img_224: Image.Image, cam01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = np.asarray(img_224.convert("RGB")).astype(np.float32)
    cam = (cam01 * 255).astype(np.uint8)

    heat = np.zeros_like(img)
    heat[..., 0] = cam
    heat[..., 1] = (255 - np.abs(cam - 128) * 2).clip(0, 255)
    heat[..., 2] = (255 - cam)

    out = (img * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)
