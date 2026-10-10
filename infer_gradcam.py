"""
Grad-CAM for PetroglyphFusionNet.

Maps are computed on the ResNet-50 branch (res_layer4) and, when present, the
highest-resolution FPN conv block. Heatmaps are resized back onto the original
image (letterbox padding is removed when letterbox training is enabled).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from rockart_config import load_config, outputs_dir
from rockart_pipeline import (
    MODEL_INPUT_SIZE,
    get_inference_transform,
    letterbox_params,
    load_petroglyphnet,
    open_rgb,
)


def gradcam(
    model: torch.nn.Module, img: torch.Tensor, target_layer: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_module, _inp, out):
        feat = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(feat, dict):
            key = sorted(feat.keys(), key=lambda k: str(k))[0]
            feat = feat[key]
        activations.append(feat)

    def full_bwd_hook(_module, _grad_in, grad_out):
        g = grad_out[0] if grad_out else None
        if g is None:
            return
        if isinstance(g, dict):
            key = sorted(g.keys(), key=lambda k: str(k))[0]
            g = g[key]
        gradients.append(g)

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(full_bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(img)
        score = out.max(dim=1).values.sum()
        score.backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")
        a = activations[0]
        g = gradients[0]
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = (weights * a).sum(dim=1, keepdim=False).relu()
        cam = cam / (cam.amax() + 1e-8)
        return cam, out
    finally:
        h1.remove()
        h2.remove()


def cam_to_original(
    cam_hw: np.ndarray,
    orig_w: int,
    orig_h: int,
    *,
    use_letterbox: bool,
    size: int = MODEL_INPUT_SIZE,
) -> np.ndarray:
    cam_hw = cv2.resize(cam_hw, (size, size), interpolation=cv2.INTER_LINEAR)
    if not use_letterbox:
        return cv2.resize(cam_hw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    nw, nh, x0, y0 = letterbox_params(orig_w, orig_h, size)
    cropped = cam_hw[y0 : y0 + nh, x0 : x0 + nw]
    if cropped.size == 0:
        return cv2.resize(cam_hw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def overlay_cam_on_image(
    pil_rgb: Image.Image,
    cam_hw: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """cam_hw already at original H×W. Returns BGR uint8."""
    rgb = np.array(pil_rgb.convert("RGB"))
    heat = cv2.applyColorMap((np.clip(cam_hw, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    if heat.shape[:2] != rgb.shape[:2]:
        heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def _resnet_target(model) -> Optional[torch.nn.Module]:
    if hasattr(model, "res_layer4"):
        return model.res_layer4[-1]
    return None


def _fpn_target(model) -> Optional[torch.nn.Module]:
    enc = getattr(model, "fpn_encoder", None)
    if enc is None:
        return None
    return enc.fpn_target_layer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM with petroglyphnet.pth")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output path prefix")
    parser.add_argument(
        "--layer",
        type=str,
        default="both",
        choices=["resnet", "fpn", "both"],
        help="Which branch to visualize",
    )
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_petroglyphnet(device=device)
    model.eval()

    tfm = get_inference_transform()
    pil = open_rgb(args.image)
    orig_w, orig_h = pil.size
    x = tfm(pil).unsqueeze(0).to(device)
    use_letterbox = bool(cfg.train.use_letterbox)

    targets = []
    if args.layer in ("resnet", "both"):
        t = _resnet_target(model)
        if t is not None:
            targets.append(("resnet", t))
    if args.layer in ("fpn", "both"):
        t = _fpn_target(model)
        if t is not None:
            targets.append(("fpn", t))
    if not targets:
        raise RuntimeError("No Grad-CAM target layer for this model branch.")

    out_prefix = Path(args.out) if args.out else outputs_dir(cfg) / "gradcam" / Path(args.image).stem
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pred_name = None
    for name, layer in targets:
        cam, logits = gradcam(model, x, layer)
        pred = int(logits.argmax(dim=1).detach().item())
        pred_name = class_names[pred]
        cam_np = cam[0].detach().cpu().numpy()
        cam_orig = cam_to_original(cam_np, orig_w, orig_h, use_letterbox=use_letterbox)
        out_bgr = overlay_cam_on_image(pil, cam_orig)
        out_path = out_prefix.with_name(f"{out_prefix.name}_{name}.jpg")
        cv2.imwrite(str(out_path), out_bgr)
        print("Saved:", out_path)

    print("Predicted:", pred_name)


if __name__ == "__main__":
    main()
