"""Train PetroglyphFusionNet: freeze-then-unfreeze, mixed classification + detection loss."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from rockart_config import load_config, outputs_dir
from rockart_model import build_petroglyph_fusion, image_level_targets
from rockart_pipeline import CLASS_NAMES_JSON, get_transforms, make_loader, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights_from_counts(counts: Counter, n_cls: int, mode: str) -> torch.Tensor | None:
    freq = np.array([counts[i] for i in range(n_cls)], dtype=np.float64)
    freq = np.maximum(freq, 1.0)
    mode = (mode or "none").lower()
    if mode in ("none", "off", "false"):
        return None
    if mode == "inverse":
        w = 1.0 / freq
    elif mode == "balanced":
        w = freq.sum() / (n_cls * freq)
    else:
        beta = 0.9999
        effective = 1.0 - np.power(beta, freq)
        w = (1.0 - beta) / np.maximum(effective, 1e-8)
    w = w / w.sum() * n_cls
    return torch.tensor(w, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Train PetroglyphFusionNet")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, help="Parent of train/ and val/ if split roots unset")
    p.add_argument("--train-dataset", type=str, default=None, help="ImageFolder train root")
    p.add_argument("--val-dataset", type=str, default=None, help="ImageFolder val root")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--branch", type=str, default=None, choices=["resnet", "fpn", "fusion"])
    p.add_argument("--fusion", type=str, default=None, choices=["concat", "gated"])
    p.add_argument("--freeze-epochs", type=int, default=None)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--no-letterbox", action="store_true")
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument("--weights", type=str, default=None)
    args = p.parse_args()
    args._cfg = load_config(args.config) if args.config else cfg
    return args


def _amp(device: torch.device, enabled: bool):
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        ctx = lambda: torch.amp.autocast("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        ctx = lambda: torch.cuda.amp.autocast(enabled=enabled)
    return scaler, ctx


def _batch_loss(model, criterion, x, y, *, has_detector: bool, det_w: float, device):
    if not has_detector:
        return criterion(model(x), y)
    targets = image_level_targets(y, int(x.shape[-2]), int(x.shape[-1]), device)
    logits, det_losses = model(x, targets=targets, return_losses=True)
    loss = criterion(logits, y)
    if det_losses:
        loss = loss + det_w * sum(det_losses.values())
    return loss


@torch.no_grad()
def _val_accuracy(model, loader, device, autocast_ctx) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx():
            pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    cfg = args._cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(args.seed if args.seed is not None else cfg.train.seed)
    set_seed(seed)

    dataset_root = Path(args.dataset or cfg.paths.dataset_root)
    train_dir = Path(
        args.train_dataset or getattr(cfg.paths, "train_dataset_root", None) or (dataset_root / "train")
    )
    val_dir = Path(
        args.val_dataset or getattr(cfg.paths, "val_dataset_root", None) or (dataset_root / "val")
    )
    weights_path = Path(args.weights or cfg.paths.weights)
    last_path = weights_path.with_name("petroglyphnet_last.pth")
    log_path = outputs_dir(cfg) / "train_log.csv"

    epochs = int(args.epochs if args.epochs is not None else cfg.train.epochs)
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.train.batch_size)
    accum = max(1, int(cfg.train.accum_steps))
    freeze_epochs = int(
        args.freeze_epochs if args.freeze_epochs is not None else cfg.train.freeze_epochs
    )
    branch = (args.branch or cfg.model.branch).lower()
    fusion = (args.fusion or cfg.model.fusion).lower()
    use_preprocess = bool(args.preprocess or cfg.train.use_preprocess)
    use_letterbox = bool(cfg.train.use_letterbox) and not args.no_letterbox
    use_sampler = bool(args.balanced_sampler or cfg.train.balanced_sampler)
    patience = int(cfg.train.early_stopping_patience)
    num_workers = int(cfg.train.num_workers)
    det_w = float(cfg.model.det_loss_weight)

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Need ImageFolder splits at {train_dir} and {val_dir}")

    tf_train = get_transforms(train=True, use_letterbox=use_letterbox, use_preprocess=use_preprocess)
    tf_val = get_transforms(train=False, use_letterbox=use_letterbox, use_preprocess=use_preprocess)
    train_ds, train_loader = make_loader(
        train_dir, train=True, batch_size=batch_size, transform=tf_train, num_workers=num_workers
    )
    val_ds, val_loader = make_loader(
        val_dir, train=False, batch_size=batch_size, transform=tf_val, num_workers=num_workers
    )

    class_names = list(train_ds.classes)
    CLASS_NAMES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    if list(val_ds.classes) != class_names:
        raise ValueError(f"val classes {val_ds.classes} do not match train {class_names}")

    counts = Counter(train_ds.targets)
    n_cls = len(class_names)
    print(f"Device={device}  seed={seed}  branch={branch}  fusion={fusion}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    print(f"Classes ({n_cls}): {class_names}")
    print(f"Train counts: {[int(counts[i]) for i in range(n_cls)]}")
    print(
        f"preprocess={use_preprocess}  letterbox={use_letterbox}  "
        f"freeze_epochs={freeze_epochs}  accum={accum}  sampler={use_sampler}"
    )

    if use_sampler:
        sample_w = [1.0 / max(counts[t], 1) for t in train_ds.targets]
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = build_petroglyph_fusion(
        n_cls,
        pretrained_backbones=bool(cfg.model.pretrained_backbones),
        freeze_backbones=freeze_epochs > 0,
        branch=branch,
        fusion=fusion,
        fpn_key_order=str(cfg.model.fpn_key_order),
        embed_source=str(cfg.model.embed_source),
        head_hidden=int(cfg.model.head_hidden),
        dropout=float(cfg.model.dropout),
    ).to(device)

    cw = class_weights_from_counts(counts, n_cls, str(cfg.train.class_weight))
    if cw is not None:
        cw = cw.to(device)
        print(f"Class weights ({cfg.train.class_weight}): {cw.detach().cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": float(cfg.train.lr_backbone)},
            {"params": list(model.head_parameters()), "lr": float(cfg.train.lr_head)},
        ],
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    has_detector = model.fpn_encoder is not None
    use_amp = device.type == "cuda" and not has_detector
    scaler, autocast_ctx = _amp(device, use_amp)
    print(f"RPN+RoI={has_detector}  det_loss_weight={det_w}")

    extra_meta = {
        "seed": seed,
        "use_preprocess": use_preprocess,
        "use_letterbox": use_letterbox,
        "fpn_key_order": str(cfg.model.fpn_key_order),
        "class_weight": str(cfg.train.class_weight),
        "embed_source": str(cfg.model.embed_source),
        "det_loss_weight": det_w,
    }

    best_acc = -1.0
    stale = 0
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as logf:
        writer = csv.DictWriter(
            logf, fieldnames=["epoch", "train_loss", "val_acc", "lr_head", "lr_backbone", "best"]
        )
        writer.writeheader()

        for epoch in range(epochs):
            if freeze_epochs > 0 and epoch == freeze_epochs:
                model.set_backbone_trainable(True)
                print(f"Unfroze backbones at epoch {epoch + 1}", flush=True)

            model.train()
            running = 0.0
            seen = 0
            optimizer.zero_grad(set_to_none=True)

            for step, (x, y) in enumerate(train_loader, start=1):
                x, y = x.to(device), y.to(device)
                with autocast_ctx():
                    loss = _batch_loss(
                        model, criterion, x, y,
                        has_detector=has_detector, det_w=det_w, device=device,
                    ) / accum
                scaler.scale(loss).backward()
                if step % accum == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += loss.item() * accum * x.size(0)
                seen += x.size(0)

            acc = _val_accuracy(model, val_loader, device, autocast_ctx)
            scheduler.step()

            is_best = acc >= best_acc
            if is_best:
                best_acc = acc
                stale = 0
                save_checkpoint(
                    weights_path, model, class_names, epoch=epoch, val_acc=acc, extra=extra_meta
                )
                local_best = Path(__file__).resolve().parent / "petroglyphnet.pth"
                if local_best.resolve() != weights_path.resolve():
                    save_checkpoint(
                        local_best, model, class_names, epoch=epoch, val_acc=acc, extra=extra_meta
                    )
                print(f"  saved best (val_acc={acc:.4f}) -> {weights_path}", flush=True)
            else:
                stale += 1
            save_checkpoint(
                last_path, model, class_names, epoch=epoch, val_acc=acc, extra=extra_meta
            )

            lrs = [pg["lr"] for pg in optimizer.param_groups]
            train_loss = running / max(seen, 1)
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.6f}",
                    "val_acc": f"{acc:.6f}",
                    "lr_head": f"{lrs[-1] if lrs else 0:.8f}",
                    "lr_backbone": f"{lrs[0] if lrs else 0:.8f}",
                    "best": int(is_best),
                }
            )
            logf.flush()
            print(
                f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_acc={acc:.4f}  best={best_acc:.4f}",
                flush=True,
            )
            if patience > 0 and stale >= patience:
                print(f"Early stopping after {patience} epochs without val_acc improvement.")
                break

    print(f"Done. Best val_acc={best_acc:.4f}. Weights: {weights_path}", flush=True)
    print(f"Log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
