"""
PetroglyphFusionNet: classify rock-art figures.

Branches: resnet | fpn | fusion
  resnet — ImageNet ResNet-50 global embedding
  fpn    — Faster R-CNN FPN features (classification) + RPN/RoI (detection)
  fusion — concatenate or gated-fuse ResNet-50 with FPN (or RoI) features

Classification defaults to FPN GAP so existing petroglyphnet.pth files load.
Set embed_source=roi to classify from RoI box-head embeddings (requires retrain).
ImageFolder training uses a whole-image box as weak localization ground truth.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

VALID_BRANCHES = ("resnet", "fpn", "fusion")
VALID_FUSIONS = ("concat", "gated")
VALID_EMBED_SOURCES = ("fpn", "roi")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Map ImageNet-normalized tensors back to RGB in [0, 1] for Faster R-CNN."""
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def fpn_sorted_keys(keys: Sequence[object], order: str = "stable") -> List[str]:
    """Concat order of FPN pyramid levels. ``stable`` = 0, 1, 2, 3, pool."""
    keys_s = [str(k) for k in keys]
    if (order or "stable").lower() == "legacy":
        prefer = ["0", "pool", "1", "2", "3"]
        rest = sorted(k for k in keys_s if k not in prefer)
        return [k for k in prefer if k in keys_s] + rest

    def _sk(k: str) -> Tuple[int, object]:
        if k.lstrip("-").isdigit():
            return (0, int(k))
        if k == "pool":
            return (1, 0)
        return (2, k)

    return sorted(keys_s, key=_sk)


def image_level_targets(
    labels: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """One box covering the tensor; label = class_idx + 1 (0 is background)."""
    x2 = float(max(width, 2))
    y2 = float(max(height, 2))
    targets: List[Dict[str, torch.Tensor]] = []
    for y in labels:
        targets.append(
            {
                "boxes": torch.tensor([[0.0, 0.0, x2, y2]], dtype=torch.float32, device=device),
                "labels": torch.tensor([int(y.item()) + 1], dtype=torch.int64, device=device),
            }
        )
    return targets


class FasterRCNNRoIEncoder(nn.Module):
    """
    Faster R-CNN ResNet-50-FPN + RPN + RoI heads.

    embed_source='fpn': GAP of each FPN level, concatenated (legacy checkpoints).
    embed_source='roi': mean of RoI box-head features (new training run).
    detect() always runs RPN + RoI.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        key_order: str = "stable",
        embed_source: str = "fpn",
    ) -> None:
        super().__init__()
        embed_source = (embed_source or "fpn").lower()
        if embed_source not in VALID_EMBED_SOURCES:
            raise ValueError(f"embed_source must be one of {VALID_EMBED_SOURCES}")

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.detector = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

        try:
            self.roi_out_dim = int(self.detector.roi_heads.box_head.fc7.out_features)
        except AttributeError:
            self.roi_out_dim = 1024

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.key_order = key_order
        self.embed_source = embed_source

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.detector.backbone(dummy)
            feat_map = {str(k): v for k, v in feats.items()}
            self.level_keys = fpn_sorted_keys(feat_map.keys(), self.key_order)
            self.fpn_out_dim = sum(int(feat_map[k].shape[1]) for k in self.level_keys)
        self.out_dim = self.roi_out_dim if embed_source == "roi" else self.fpn_out_dim

    @property
    def body(self) -> nn.Module:
        return self.detector.backbone

    def fpn_target_layer(self) -> nn.Module:
        fpn = self.detector.backbone.fpn
        if hasattr(fpn, "layer_blocks") and len(fpn.layer_blocks) > 0:
            return fpn.layer_blocks[0]
        return self.detector.backbone

    def _images_01(self, x: torch.Tensor) -> List[torch.Tensor]:
        x01 = denormalize_imagenet(x.float())
        return [x01[i] for i in range(x01.shape[0])]

    def _fpn_gap_embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)
        feat_map = {str(k): v for k, v in feats.items()}
        parts = [self.pool(feat_map[k]).flatten(1) for k in self.level_keys]
        return torch.cat(parts, dim=1)

    def _run_rpn(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        images, targets_t = self.detector.transform(self._images_01(x), targets)
        features = self.detector.backbone(images.tensors)
        proposals, proposal_losses = self.detector.rpn(images, features, targets_t)
        return images, targets_t, features, proposals, proposal_losses

    def _pool_roi_features(
        self,
        box_features: torch.Tensor,
        proposals: List[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        embeds = []
        offset = 0
        dim = self.roi_out_dim
        dtype = box_features.dtype
        for i in range(batch_size):
            n = int(proposals[i].shape[0]) if i < len(proposals) else 0
            if n <= 0 or box_features.numel() == 0 or offset >= box_features.shape[0]:
                embeds.append(torch.zeros(dim, device=device, dtype=dtype))
            else:
                embeds.append(box_features[offset : offset + n].mean(dim=0))
            offset += n
        return torch.stack(embeds, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        det_losses: Dict[str, torch.Tensor] = {}
        need_rpn = self.embed_source == "roi" or (self.training and targets is not None)
        images = targets_t = features = proposals = proposal_losses = None
        if need_rpn:
            images, targets_t, features, proposals, proposal_losses = self._run_rpn(x, targets)

        if self.embed_source == "roi":
            assert images is not None and features is not None and proposals is not None
            box_features = self.detector.roi_heads.box_roi_pool(
                features, proposals, images.image_sizes
            )
            box_features = self.detector.roi_heads.box_head(box_features)
            embed = self._pool_roi_features(box_features, proposals, x.shape[0], x.device)
        else:
            embed = self._fpn_gap_embed(x)

        if self.training and targets_t is not None:
            assert images is not None and features is not None and proposals is not None
            _, detector_losses = self.detector.roi_heads(
                features, proposals, images.image_sizes, targets_t
            )
            if proposal_losses:
                det_losses.update(proposal_losses)
            det_losses.update(detector_losses)
        return embed, det_losses

    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        was_training = self.detector.training
        self.detector.eval()
        try:
            return self.detector(self._images_01(x))
        finally:
            self.detector.train(was_training)


class GatedFusionHead(nn.Module):
    """Soft-gated blend of projected ResNet and FPN/RoI embeddings."""

    def __init__(
        self,
        dim_r: int,
        dim_f: int,
        hidden: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj_r = nn.Linear(dim_r, hidden)
        self.proj_f = nn.Linear(dim_f, hidden)
        self.gate = nn.Sequential(
            nn.Linear(dim_r + dim_f, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.cls = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        g = torch.softmax(self.gate(torch.cat([r, f], dim=1)), dim=1)
        z = g[:, 0:1] * self.proj_r(r) + g[:, 1:2] * self.proj_f(f)
        return self.cls(z)


def _is_head_param(name: str) -> bool:
    return name.startswith("head.") or name.startswith("gated.")


class PetroglyphFusionNet(nn.Module):
    """Fuse ResNet-50 appearance with Faster R-CNN FPN/RoI features, then classify."""

    def __init__(
        self,
        num_classes: int,
        pretrained_backbones: bool = True,
        freeze_backbones: bool = False,
        head_hidden: int = 512,
        dropout: float = 0.3,
        branch: str = "fusion",
        fusion: str = "concat",
        fpn_key_order: str = "stable",
        embed_source: str = "fpn",
    ) -> None:
        super().__init__()
        branch = branch.lower()
        fusion = fusion.lower()
        embed_source = (embed_source or "fpn").lower()
        if branch not in VALID_BRANCHES:
            raise ValueError(f"branch must be one of {VALID_BRANCHES}, got {branch!r}")
        if fusion not in VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {VALID_FUSIONS}, got {fusion!r}")
        if embed_source not in VALID_EMBED_SOURCES:
            raise ValueError(
                f"embed_source must be one of {VALID_EMBED_SOURCES}, got {embed_source!r}"
            )

        self.num_classes = num_classes
        self.branch = branch
        self.fusion = fusion
        self.fpn_key_order = fpn_key_order
        self.embed_source = embed_source
        self._res_dim = 2048
        self.fpn_encoder: FasterRCNNRoIEncoder | None = None
        self.gated: GatedFusionHead | None = None
        self.head: nn.Sequential | None = None

        if branch in ("resnet", "fusion"):
            w = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbones else None
            res = resnet50(weights=w)
            self.res_stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
            self.res_layer1 = res.layer1
            self.res_layer2 = res.layer2
            self.res_layer3 = res.layer3
            self.res_layer4 = res.layer4
            self.res_gap = nn.AdaptiveAvgPool2d(1)

        if branch in ("fpn", "fusion"):
            self.fpn_encoder = FasterRCNNRoIEncoder(
                num_classes=num_classes,
                pretrained=pretrained_backbones,
                key_order=fpn_key_order,
                embed_source=embed_source,
            )

        if branch == "resnet":
            in_dim = self._res_dim
        elif branch == "fpn":
            in_dim = self.fpn_encoder.out_dim  # type: ignore[union-attr]
        else:
            in_dim = self._res_dim + self.fpn_encoder.out_dim  # type: ignore[union-attr]

        if branch == "fusion" and fusion == "gated":
            self.gated = GatedFusionHead(
                self._res_dim,
                self.fpn_encoder.out_dim,  # type: ignore[union-attr]
                head_hidden,
                num_classes,
                dropout,
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )

        if freeze_backbones:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, p in self.named_parameters():
            if _is_head_param(name):
                continue
            if ".rpn." in name or ".roi_heads." in name:
                p.requires_grad = True
                continue
            p.requires_grad = trainable

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        for name, p in self.named_parameters():
            if not _is_head_param(name):
                yield p

    def head_parameters(self) -> Iterator[nn.Parameter]:
        for name, p in self.named_parameters():
            if _is_head_param(name):
                yield p

    def _resnet_embed(self, x: torch.Tensor) -> torch.Tensor:
        r = self.res_stem(x)
        r = self.res_layer1(r)
        r = self.res_layer2(r)
        r = self.res_layer3(r)
        r = self.res_layer4(r)
        return self.res_gap(r).flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_losses: bool = False,
    ):
        det_losses: Dict[str, torch.Tensor] = {}
        if self.branch == "resnet":
            logits = self.head(self._resnet_embed(x))  # type: ignore[misc]
        elif self.branch == "fpn":
            z, det_losses = self.fpn_encoder(x, targets)  # type: ignore[misc]
            logits = self.head(z)  # type: ignore[misc]
        else:
            r = self._resnet_embed(x)
            f, det_losses = self.fpn_encoder(x, targets)  # type: ignore[misc]
            if self.gated is not None:
                logits = self.gated(r, f)
            else:
                logits = self.head(torch.cat([r, f], dim=1))  # type: ignore[misc]
        if return_losses:
            return logits, det_losses
        return logits

    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        if self.fpn_encoder is None:
            raise RuntimeError("detect() requires branch 'fpn' or 'fusion'")
        return self.fpn_encoder.detect(x)


def build_petroglyph_fusion(num_classes: int, **kwargs) -> PetroglyphFusionNet:
    return PetroglyphFusionNet(num_classes, **kwargs)
