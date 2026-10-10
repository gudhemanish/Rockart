"""
Rock-art data utilities: EXIF-aware ImageFolder, letterbox, optional enhance,
checkpoint load/save, and inference helpers aligned with ImageNet normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rockart_config import PACKAGE_DIR, load_config
from rockart_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PetroglyphFusionNet,
    build_petroglyph_fusion,
)

Image.MAX_IMAGE_PIXELS = None

cfg = load_config()

PETROGLYPHNET_WEIGHTS = Path(cfg.paths.weights)
CLASS_NAMES_JSON = PACKAGE_DIR / "class_names.json"

IMAGENET_FILL = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)
MODEL_INPUT_SIZE = int(cfg.model.input_size)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def weight_candidates() -> List[Path]:
    primary = PETROGLYPHNET_WEIGHTS
    return [
        primary,
        PACKAGE_DIR / "petroglyphnet.pth",
        PACKAGE_DIR / "outputs" / "petroglyphnet.pth",
        primary.with_name("petroglyphnet_last.pth"),
        PACKAGE_DIR / "petroglyphnet_last.pth",
        PACKAGE_DIR / "outputs" / "petroglyphnet_last.pth",
    ]


def resolve_petroglyphnet_weights() -> Path:
    for path in weight_candidates():
        if path.is_file():
            return path
    return PETROGLYPHNET_WEIGHTS


def _torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(
    path: Union[str, Path],
    model: PetroglyphFusionNet,
    class_names: Sequence[str],
    *,
    epoch: int,
    val_acc: float,
    extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "epoch": epoch,
        "val_acc": val_acc,
        "branch": getattr(model, "branch", "fusion"),
        "fusion": getattr(model, "fusion", "concat"),
        "fpn_key_order": getattr(model, "fpn_key_order", "stable"),
        "embed_source": getattr(model, "embed_source", "fpn"),
        "num_classes": len(class_names),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def migrate_legacy_fpn_state_dict(state_dict: dict) -> Tuple[dict, bool]:
    """Map ``fpn_encoder.body.*`` onto ``fpn_encoder.detector.backbone.*``."""
    is_legacy = any(k.startswith("fpn_encoder.body.") for k in state_dict)
    if not is_legacy:
        return state_dict, False
    migrated = {}
    prefix = "fpn_encoder.body."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            migrated["fpn_encoder.detector.backbone." + key[len(prefix) :]] = value
        else:
            migrated[key] = value
    return migrated, True


def _infer_embed_source(state_dict: dict, fallback: str) -> str:
    weight = state_dict.get("head.0.weight")
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return fallback
    in_features = int(weight.shape[1])
    if in_features == 3328:
        return "fpn"
    if in_features == 3072:
        return "roi"
    return fallback


def load_petroglyphnet(
    weights_path: Optional[Union[Path, str]] = None,
    class_names_path: Optional[Union[Path, str]] = None,
    device: Optional[torch.device] = None,
    *,
    branch: Optional[str] = None,
    fusion: Optional[str] = None,
) -> Tuple[PetroglyphFusionNet, List[str]]:
    """Load PetroglyphFusionNet. Older FPN-backbone checkpoints are remapped."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wpath = Path(weights_path) if weights_path else resolve_petroglyphnet_weights()
    if not wpath.is_file():
        searched = "\n  ".join(str(p) for p in weight_candidates())
        raise FileNotFoundError(
            f"petroglyphnet.pth not found (looked in):\n  {searched}\n"
            "Train first: python train_rockart_model.py"
        )

    raw = _torch_load_checkpoint(wpath, device)
    meta: dict = {}
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        meta = raw
    else:
        state_dict = raw

    if isinstance(meta.get("class_names"), list):
        class_names = list(meta["class_names"])
    else:
        cpath = Path(class_names_path) if class_names_path else CLASS_NAMES_JSON
        if not cpath.is_file():
            raise FileNotFoundError(f"class_names.json not found at {cpath}")
        with open(cpath, encoding="utf-8") as f:
            class_names = json.load(f)

    branch = branch or str(meta.get("branch") or cfg.model.branch)
    fusion = fusion or str(meta.get("fusion") or cfg.model.fusion)
    fpn_key_order = str(meta.get("fpn_key_order") or cfg.model.fpn_key_order)
    state_dict, is_legacy = migrate_legacy_fpn_state_dict(state_dict)
    embed_source = _infer_embed_source(
        state_dict, str(meta.get("embed_source") or cfg.model.embed_source)
    )

    model = build_petroglyph_fusion(
        len(class_names),
        pretrained_backbones=is_legacy,
        branch=branch,
        fusion=fusion,
        fpn_key_order=fpn_key_order,
        embed_source=embed_source,
        head_hidden=int(cfg.model.head_hidden),
        dropout=float(cfg.model.dropout),
    )
    if is_legacy:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        missing_core = [k for k in missing if ".rpn." not in k and ".roi_heads." not in k]
        if missing_core or unexpected:
            raise RuntimeError(
                "Could not migrate the older FPN checkpoint onto Faster R-CNN. "
                f"Missing core keys: {missing_core[:8]} Unexpected: {unexpected[:8]}"
            )
        model._rpn_roi_from_coco = True  # type: ignore[attr-defined]
        print(
            "Loaded older FPN checkpoint into Faster R-CNN: backbone + classifier "
            "restored; RPN and RoI heads initialized from COCO. "
            "For rock-art mAP, run: python train_rockart_model.py"
        )
    else:
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                "Checkpoint does not match the current model. "
                "Retrain with train_rockart_model.py."
            ) from e
        model._rpn_roi_from_coco = False  # type: ignore[attr-defined]
    model.to(device)
    model.eval()
    return model, class_names


def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def lab_l_equalize(img_rgb: np.ndarray) -> np.ndarray:
    """Equalize Lab L-channel (contrast stretch). Not PCA/DStretch."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    merged = cv2.merge((cv2.equalizeHist(l_ch), a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_Lab2RGB)


def preprocess_array(img_rgb: np.ndarray) -> np.ndarray:
    """Denoise + Lab L-eq. Returns uint8 RGB."""
    return lab_l_equalize(denoise_image(img_rgb))


def letterbox_params(
    orig_w: int, orig_h: int, size: int = MODEL_INPUT_SIZE
) -> Tuple[int, int, int, int]:
    """Return (nw, nh, x0, y0) of the pasted region inside a size×size canvas."""
    scale = size / max(orig_w, orig_h)
    nw = max(1, int(round(orig_w * scale)))
    nh = max(1, int(round(orig_h * scale)))
    return nw, nh, (size - nw) // 2, (size - nh) // 2


class LetterboxResize:
    """Pad to square (ImageNet mean fill) then resize — preserves aspect ratio."""

    def __init__(self, size: int = MODEL_INPUT_SIZE, fill: Tuple[int, int, int] = IMAGENET_FILL):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Empty image")
        nw, nh, x0, y0 = letterbox_params(w, h, self.size)
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, (x0, y0))
        return canvas


class RockartEnhance:
    """Optional denoise + Lab L-eq as a PIL transform. Use in train AND eval together."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return Image.fromarray(preprocess_array(np.array(img.convert("RGB"))))


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def require_imagefolder(root: Union[str, Path]) -> Path:
    """Ensure root has class subfolders with images (ImageFolder layout)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    class_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if class_dirs:
        return root
    n_img = sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(
        f"No class folders in {root}. Training needs ImageFolder layout:\n"
        f"  {root}\\<class_name>\\image.jpg\n"
        f"  e.g. Buffalo, Deer, Elephant, Horse, Man\n"
        f"This folder has {n_img} image(s) in the root (unlabeled). "
        "Use a labeled split, or move each image into its class subfolder."
    )


class ExifImageFolder(datasets.ImageFolder):
    """ImageFolder that applies EXIF transpose so camera rotation matches inference."""

    def __init__(self, root: str, transform=None, target_transform=None):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=_is_image_file,
        )

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def _train_augments() -> list:
    return [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
    ]


def _geometry_transform(size: int, use_letterbox: bool, train: bool) -> list:
    if use_letterbox:
        geo: list = [LetterboxResize(size)]
        if train:
            geo.extend(_train_augments())
        return geo
    if train:
        return [
            transforms.Resize(int(size * 1.15)),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            *_train_augments(),
        ]
    return [transforms.Resize(int(size * 1.15)), transforms.CenterCrop(size)]


def get_transforms(
    *,
    train: bool = False,
    size: int = MODEL_INPUT_SIZE,
    use_letterbox: Optional[bool] = None,
    use_preprocess: Optional[bool] = None,
) -> transforms.Compose:
    if use_letterbox is None:
        use_letterbox = bool(cfg.train.use_letterbox)
    if use_preprocess is None:
        use_preprocess = bool(cfg.train.use_preprocess)

    steps: list = []
    if use_preprocess:
        steps.append(RockartEnhance())
    steps.extend(_geometry_transform(size, use_letterbox, train))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
    )
    return transforms.Compose(steps)


def get_inference_transform(size: int = MODEL_INPUT_SIZE) -> transforms.Compose:
    return get_transforms(train=False, size=size)


def open_rgb(path: Union[str, Path]) -> Image.Image:
    with Image.open(path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def folder_idx_to_model_idx(
    folder_classes: Sequence[str],
    class_names: Sequence[str],
) -> np.ndarray:
    """Map ImageFolder integer labels onto checkpoint class order."""
    name_to_model = {n: i for i, n in enumerate(class_names)}
    unknown = [c for c in folder_classes if c not in name_to_model]
    if unknown:
        raise ValueError(
            f"Split folders {list(folder_classes)} contain classes not in the checkpoint "
            f"{list(class_names)}: {unknown}"
        )
    return np.array([name_to_model[c] for c in folder_classes], dtype=np.int64)


def make_loader(
    root: Union[str, Path],
    *,
    train: bool = False,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    sampler=None,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[ExifImageFolder, DataLoader]:
    root = require_imagefolder(root)
    tfm = transform if transform is not None else get_transforms(train=train)
    ds = ExifImageFolder(str(root), transform=tfm)
    if shuffle is None:
        shuffle = train and sampler is None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


def iter_image_paths(root: Union[str, Path], *, unlabeled: bool = False):
    """Yield (path, true_class_or_None). ImageFolder class subfolders, or a flat folder."""
    root = Path(root)

    def _files(folder: Path):
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p

    if unlabeled:
        for p in _files(root):
            yield p, None
        return
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        for p in _files(root):
            yield p, None
        return
    for cls_dir in subdirs:
        for p in _files(cls_dir):
            yield p, cls_dir.name
