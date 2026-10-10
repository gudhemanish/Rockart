"""Load config.yaml with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Union

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "paths": {
        "dataset_root": r"D:/Sunayana/PHD/Project/RockArt_Images/Datasets_Decorrelation_Stretch",
        "train_dataset_root": r"D:/Sunayana/PHD/Project/RockArt_Images/Datasets_Decorrelation_Stretch/train",
        "val_dataset_root": r"D:/Sunayana/PHD/Project/RockArt_Images/Datasets_Decorrelation_Stretch/val",
        "weights": r"D:/Sunayana/PHD/Project/CNN/Model/petroglyphnet.pth",
        "unlabeled_dir": r"D:/Sunayana/PHD/Project/CNN/Datasets/Train",
        "outputs_dir": "outputs",
    },
    "model": {
        "input_size": 224,
        "branch": "fusion",
        "fusion": "concat",
        "fpn_key_order": "stable",
        "embed_source": "fpn",
        "head_hidden": 512,
        "dropout": 0.3,
        "pretrained_backbones": True,
        "det_loss_weight": 0.5,
    },
    "train": {
        "seed": 42,
        "epochs": 30,
        "batch_size": 4,
        "accum_steps": 2,
        "lr_head": 3.0e-4,
        "lr_backbone": 3.0e-5,
        "weight_decay": 1.0e-4,
        "freeze_epochs": 5,
        "early_stopping_patience": 7,
        "num_workers": 0,
        "use_preprocess": False,
        "use_letterbox": True,
        "balanced_sampler": False,
        "class_weight": "effective",
    },
    "eval": {
        "batch_size": 8,
        "with_ttest": False,
        "n_bootstrap": 1000,
        "alpha": 0.05,
        "score_thresh": 0.05,
        "iou_start": 0.50,
        "iou_end": 0.95,
        "iou_step": 0.05,
    },
}


def _deep_merge(base: dict, overlay: Mapping) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in {path}. Use forward slashes in quoted paths, "
                r'e.g. "D:/Sunayana/PHD/Project/CNN/Datasets" not "D:\Sunayana\...".'
            ) from e
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_config(path: Optional[Union[str, Path]] = None) -> SimpleNamespace:
    cfg_path = Path(path) if path else Path(os.environ.get("ROCKART_CONFIG", DEFAULT_CONFIG_PATH))
    merged = _deep_merge(_DEFAULTS, _load_yaml(cfg_path))

    env_dataset = os.environ.get("ROCKART_DATASET_ROOT")
    env_train = os.environ.get("ROCKART_TRAIN_DATASET_ROOT")
    env_val = os.environ.get("ROCKART_VAL_DATASET_ROOT")
    env_weights = os.environ.get("ROCKART_WEIGHTS")
    env_unlabeled = os.environ.get("ROCKART_UNLABELED_DIR")
    if env_dataset:
        merged["paths"]["dataset_root"] = env_dataset
    if env_train:
        merged["paths"]["train_dataset_root"] = env_train
    if env_val:
        merged["paths"]["val_dataset_root"] = env_val
    if env_weights:
        merged["paths"]["weights"] = env_weights
    if env_unlabeled:
        merged["paths"]["unlabeled_dir"] = env_unlabeled

    ns = _to_ns(merged)
    ns._config_path = str(cfg_path)
    ns._dict = merged
    return ns


def outputs_dir(cfg: Optional[SimpleNamespace] = None) -> Path:
    cfg = cfg or load_config()
    raw = Path(cfg.paths.outputs_dir)
    out = raw if raw.is_absolute() else PACKAGE_DIR / raw
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_dir(split: str, cfg: Optional[SimpleNamespace] = None, dataset_root: Optional[str] = None) -> Path:
    """ImageFolder root for a split. Train/val use dedicated roots when set."""
    cfg = cfg or load_config()
    if dataset_root:
        return Path(dataset_root) / split
    dedicated = {
        "train": getattr(cfg.paths, "train_dataset_root", None),
        "val": getattr(cfg.paths, "val_dataset_root", None),
    }.get(split)
    if dedicated:
        return Path(dedicated)
    return Path(cfg.paths.dataset_root) / split
