"""
Batch inference on an image folder using petroglyphnet.pth, write predictions.csv,
then evaluate ImageFolder train/val/test (same dataset_root and transforms as training).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from evaluate_metrics import evaluate_metrics
from rockart_config import load_config, outputs_dir
from rockart_pipeline import (
    get_inference_transform,
    iter_image_paths,
    load_petroglyphnet,
    open_rgb,
)

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Predict rock-art images and evaluate splits")
    p.add_argument(
        "--images",
        type=str,
        default=str(Path(cfg.paths.dataset_root) / "test"),
        help="ImageFolder test dir (class subfolders) or a flat folder with --unlabeled",
    )
    p.add_argument(
        "--unlabeled",
        action="store_true",
        help="Treat --images as a flat folder (e.g. site photos without class subfolders)",
    )
    p.add_argument(
        "--unlabeled-dir",
        type=str,
        default=None,
        help="Shortcut: predict cfg unlabeled_dir as a flat folder",
    )
    p.add_argument("--csv", type=str, default=None, help="Predictions CSV path")
    p.add_argument("--skip-eval", action="store_true", help="Skip train/val/test metrics")
    p.add_argument("--ttest", action="store_true", help="Include appendix t-tests in eval")
    p.add_argument("--dataset", type=str, default=None, help="Dataset root for metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_petroglyphnet(device=device)
    transform = get_inference_transform()

    if args.unlabeled_dir:
        image_root = Path(args.unlabeled_dir)
        unlabeled = True
    else:
        image_root = Path(args.images)
        unlabeled = bool(args.unlabeled)

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_root}")

    csv_path = Path(args.csv) if args.csv else outputs_dir(cfg) / "predictions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f" PREDICTIONS — {image_root}")
    print("=" * 72 + "\n")

    rows = []
    for img_path, true_cls in iter_image_paths(image_root, unlabeled=unlabeled):
        img = open_rgb(img_path)
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0]
            pred = int(out.argmax(1).item())

        top_p, top_i = probs.topk(min(3, len(class_names)))
        top_str = ", ".join(
            f"{class_names[i]} {p:.3f}" for p, i in zip(top_p.tolist(), top_i.tolist())
        )
        pred_name = class_names[pred]
        print(f"{img_path.name} -> {pred_name}  (top: {top_str})")
        row = {
            "filepath": str(img_path),
            "filename": img_path.name,
            "true_class": true_cls or "",
            "pred_class": pred_name,
            "pred_index": pred,
            "correct": "" if true_cls is None else int(true_cls == pred_name),
        }
        for i, name in enumerate(class_names):
            row[f"p_{name}"] = f"{float(probs[i]):.6f}"
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} predictions -> {csv_path}")
    else:
        print("No images found.")

    if args.skip_eval:
        return

    evaluate_metrics(
        model=model,
        class_names=class_names,
        device=device,
        with_ttest=args.ttest,
        dataset_root=args.dataset,
    )


if __name__ == "__main__":
    main()
