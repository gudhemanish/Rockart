"""
Evaluation for PetroglyphFusionNet.

Classification: accuracy (+ bootstrap CI), precision, recall, F1, confusion matrix.
Faster R-CNN RPN+RoI: mAP@0.5 and mAP@[0.5:0.95] (weak whole-image boxes).
Binomial test vs chance (preferred). Optional appendix t-tests.
Examiner-style train vs held-out overfitting check.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from rockart_config import load_config, outputs_dir, split_dir
from rockart_model import image_level_targets
from rockart_pipeline import (
    folder_idx_to_model_idx,
    get_transforms,
    load_petroglyphnet,
    make_loader,
)

cfg = load_config()

OVERFIT_GAP = 0.12
OVERFIT_TRAIN_HIGH = 0.78
SIMILAR_GAP = 0.08
LOW_ACC_BOTH = 0.52


def multiclass_ovr_tp_fp_fn_tn(
    cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    total = cm.sum()
    tn = total - (tp + fn + fp)
    return tp, fp, fn, tn


def per_class_precision_recall_f1(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def print_ovr_binary_tables(
    class_names: List[str],
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
) -> None:
    print("\nOne-vs-rest binary tables (each row = that class treated as positive):\n")
    for i, name in enumerate(class_names):
        print(f"--- Class: {name!r} (positive) ---")
        print(
            f"{'':24} {'Pred Positive':>16} {'Pred Negative':>16}\n"
            f"{'Actual Positive':24} {int(tp[i]):>16} {int(fn[i]):>16}\n"
            f"{'Actual Negative':24} {int(fp[i]):>16} {int(tn[i]):>16}\n"
        )


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    correct = (y_true == y_pred).astype(np.float64)
    accs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = correct[idx].mean()
    lo, hi = np.quantile(accs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(correct.mean()), float(lo), float(hi)


def binomial_vs_chance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    split_label: str,
    alpha: float = 0.05,
) -> dict:
    k = int((np.asarray(y_true) == np.asarray(y_pred)).sum())
    n = int(len(y_true))
    p_chance = 1.0 / n_classes
    res = stats.binomtest(k, n, p=p_chance, alternative="greater")
    pval = float(res.pvalue)
    print(
        f"\n--- Binomial test vs chance [{split_label}] ---\n"
        f"  Correct {k}/{n}  chance={p_chance:.4f}  p={pval:.4g}  (H1: better than chance)\n"
        f"  Verdict: {'SIGNIFICANT [OK]' if pval < alpha else 'NOT SIGNIFICANT [X]'} "
        f"at alpha={alpha}\n"
    )
    return {
        "split": split_label,
        "k": k,
        "n": n,
        "p_chance": p_chance,
        "p_value": pval,
        "significant": pval < alpha,
    }


def run_ttest_vs_chance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    split_label: str,
    alpha: float = 0.05,
) -> dict:
    chance = 1.0 / n_classes
    correct = (y_true == y_pred).astype(np.float64)
    t_stat, p_two = stats.ttest_1samp(correct, popmean=chance)
    p_one = float(p_two) / 2.0
    print(
        f"\n--- Appendix t-test vs chance [{split_label}] ---\n"
        f"  Limitation: samples are not independent (same site/panel).\n"
        f"  mean={correct.mean():.4f}  t={t_stat:.4f}  p_one={p_one:.4g}\n"
    )
    return {
        "split": split_label,
        "chance": chance,
        "sample_mean": float(correct.mean()),
        "t_stat": float(t_stat),
        "p_two_tail": float(p_two),
        "p_one_tail": p_one,
        "significant_above_chance": bool(t_stat > 0 and p_one < alpha),
    }


def run_ttest_two_splits(results_a: dict, results_b: dict, alpha: float = 0.05) -> dict:
    label_a = results_a["split_label"]
    label_b = results_b["split_label"]
    correct_a = (results_a["y_true"] == results_a["y_pred"]).astype(np.float64)
    correct_b = (results_b["y_true"] == results_b["y_pred"]).astype(np.float64)
    t_stat, p_val = stats.ttest_ind(correct_a, correct_b, equal_var=False)
    print(
        f"\n--- Appendix Welch t-test: {label_a} vs {label_b} ---\n"
        f"  mean {label_a}={correct_a.mean():.4f}  {label_b}={correct_b.mean():.4f}  "
        f"p={float(p_val):.4g}\n"
        "  Limitation: splits are not independent random samples from one population.\n"
    )
    return {
        "split_a": label_a,
        "split_b": label_b,
        "mean_a": float(correct_a.mean()),
        "mean_b": float(correct_b.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < alpha),
    }


def print_ttest_report(
    split_predictions: Dict[str, dict],
    n_classes: int,
    alpha: float = 0.05,
) -> Dict[str, list]:
    print("\n" + "=" * 72)
    print(" APPENDIX: T-TESTS (interpret with caution)")
    print("=" * 72)
    print(
        "Binary correctness is not i.i.d. for related rock-art photos.\n"
        "Prefer binomial-vs-chance and bootstrap CIs in the main text.\n"
    )
    one_sample_results = []
    for name, preds in split_predictions.items():
        one_sample_results.append(
            run_ttest_vs_chance(preds["y_true"], preds["y_pred"], n_classes, name, alpha)
        )
    pairwise_results = []
    names = list(split_predictions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise_results.append(
                run_ttest_two_splits(split_predictions[names[i]], split_predictions[names[j]], alpha)
            )
    return {"one_sample": one_sample_results, "pairwise": pairwise_results}


def print_examiner_validation_overfitting_report(
    accuracies: Dict[str, Optional[float]],
) -> None:
    print("\n" + "=" * 72)
    print(" VALIDATION OF RESULTS - Overfitting check (examiner view)")
    print("=" * 72)
    print(
        f"""
  High train + low held-out  -> overfitting
  Both low                   -> underfitting
  Similar train and held-out -> generalizes

Heuristics: high train>={OVERFIT_TRAIN_HIGH:.2f}, gap>={OVERFIT_GAP:.2f}, """
        f"low both<{LOW_ACC_BOTH:.2f}, similar |gap|<={SIMILAR_GAP:.2f}"
    )

    train_acc = accuracies.get("train")
    test_acc = accuracies.get("test")
    val_acc = accuracies.get("val")

    held_out_label = "test"
    held_out_acc = test_acc
    if held_out_acc is None and val_acc is not None:
        held_out_label = "val (proxy; test not evaluated)"
        held_out_acc = val_acc

    print("Accuracy summary used for diagnosis:")
    print(f"  train: {train_acc if train_acc is not None else 'n/a'}")
    print(f"  val:   {val_acc if val_acc is not None else 'n/a'}")
    print(f"  test:  {test_acc if test_acc is not None else 'n/a'}")
    print(f"  Held-out for train comparison: {held_out_label} -> {held_out_acc}")

    if train_acc is None or held_out_acc is None:
        print(
            "\nDiagnosis: INCOMPLETE — need both train and at least one held-out split.\n"
        )
        return

    gap = train_acc - held_out_acc
    overfit = (
        train_acc >= OVERFIT_TRAIN_HIGH
        and gap >= OVERFIT_GAP
        and held_out_acc < train_acc - OVERFIT_GAP * 0.5
    )
    underfit = train_acc < LOW_ACC_BOTH and held_out_acc < LOW_ACC_BOTH
    good = (
        abs(gap) <= SIMILAR_GAP
        and train_acc >= LOW_ACC_BOTH
        and held_out_acc >= LOW_ACC_BOTH
    )

    print("\n--- Diagnosis ---\n")
    if overfit:
        print(
            "Verdict: LIKELY OVERFITTING [X]\n"
            f"  Train ({train_acc:.4f}) vs {held_out_label} ({held_out_acc:.4f}); gap={gap:+.4f}."
        )
    elif underfit:
        print(
            "Verdict: LIKELY UNDERFITTING [X]\n"
            f"  Both train ({train_acc:.4f}) and {held_out_label} ({held_out_acc:.4f}) are low."
        )
    elif good:
        print(
            "Verdict: GOOD GENERALIZATION [OK]\n"
            f"  Train ({train_acc:.4f}) and {held_out_label} ({held_out_acc:.4f}) "
            f"are similar (gap {gap:+.4f})."
        )
    else:
        print(
            "Verdict: MIXED / REVIEW MANUALLY\n"
            f"  Train={train_acc:.4f}, {held_out_label}={held_out_acc:.4f}, gap={gap:+.4f}."
        )
    print()


def plot_confusion_matrix(
    cm_arr: np.ndarray, classes: List[str], out_basename: str
) -> str:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (rows = actual, columns = predicted)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(
                j,
                i,
                int(cm_arr[i, j]),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )

    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_dir = outputs_dir() / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / f"{out_basename}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix figure saved: {out_file}")
    return out_file


def _eval_one_split(
    model: torch.nn.Module,
    class_names: List[str],
    dataset_path: str,
    split_label: str,
    device: torch.device,
    n_bootstrap: int,
    alpha: float,
    batch_size: int,
) -> Optional[dict]:
    if not Path(dataset_path).is_dir():
        print(f"\n[evaluate_metrics] Skip (missing folder): {dataset_path}\n")
        return None

    try:
        dataset, loader = make_loader(
            dataset_path,
            train=False,
            batch_size=batch_size,
            transform=get_transforms(train=False),
        )
    except (FileNotFoundError, OSError) as e:
        print(f"\n[evaluate_metrics] Skip {split_label!r}: {dataset_path} ({e})\n")
        return None

    if len(dataset) == 0:
        print(f"\n[evaluate_metrics] Skip {split_label!r}: no samples in {dataset_path}\n")
        return None

    try:
        idx_map = folder_idx_to_model_idx(dataset.classes, class_names)
    except ValueError as e:
        print(f"\n[evaluate_metrics] Skip {split_label!r}: {e}\n")
        return None

    if list(dataset.classes) != list(class_names):
        print(
            f"\n[evaluate_metrics] Remapping {split_label!r} folder classes "
            f"{list(dataset.classes)} -> checkpoint {list(class_names)}.\n"
        )

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            mapped = idx_map[np.asarray(labels.numpy(), dtype=np.int64)]
            y_true.extend(mapped.tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    total = int(cm.sum())
    acc, acc_lo, acc_hi = bootstrap_accuracy_ci(
        y_true_arr, y_pred_arr, n_boot=n_bootstrap, alpha=alpha
    )
    prec_macro = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    rec_macro = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    prec_weighted = precision_score(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )
    rec_weighted = recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    print(f"\n--- Split: {split_label} ---\n")
    print(f"Samples: {total}\n")
    print("--- Overall classification metrics ---\n")
    n_correct = int((y_true_arr == y_pred_arr).sum())
    print(
        f"Accuracy:           {acc:.4f}  "
        f"(95% bootstrap CI {acc_lo:.4f}–{acc_hi:.4f}; {n_correct}/{total})"
    )
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1-score (macro):  {f1_macro:.4f}")
    print()
    print(f"Precision (weighted): {prec_weighted:.4f}")
    print(f"Recall (weighted):    {rec_weighted:.4f}")
    print(f"F1-score (weighted):  {f1_weighted:.4f}")

    binomial_vs_chance(y_true_arr, y_pred_arr, len(class_names), split_label, alpha=alpha)

    tp, fp, fn, tn = multiclass_ovr_tp_fp_fn_tn(cm)
    p_ovr, r_ovr, f1_ovr = per_class_precision_recall_f1(tp, fp, fn)

    print_ovr_binary_tables(class_names, tp, fp, fn, tn)

    print("--- Per-class Precision / Recall / F1 (one-vs-rest from TP,FP,FN) ---\n")
    for i, name in enumerate(class_names):
        print(f"  {name:12}  P={p_ovr[i]:.4f}  R={r_ovr[i]:.4f}  F1={f1_ovr[i]:.4f}")

    print("\n--- Confusion matrix (counts) ---\n")
    header = " " * 14 + "".join(f"{n:>12}" for n in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>12}" for j in range(cm.shape[1]))
        print(f"{row_name:14}{row}")

    plot_confusion_matrix(cm, class_names, f"confusion_matrix_{split_label}")
    return {
        "split_label": split_label,
        "accuracy": float(acc),
        "acc_ci": (acc_lo, acc_hi),
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "dataset": dataset,
        "idx_map": idx_map,
    }


def _ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    rec = np.concatenate(([0.0], recalls, [1.0]))
    prec = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = rec >= t
        ap += float(prec[mask].max()) if np.any(mask) else 0.0
    return ap / 101.0


def _class_ap_at_iou(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    iou_thr: float,
) -> float:
    """VOC/COCO-style AP for one class at one IoU threshold. Lists are per-image."""
    n_gt = sum(len(g) for g in gt_boxes)
    if n_gt == 0:
        return float("nan")
    rows = []
    for img_i, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for b, s in zip(boxes, scores):
            rows.append((float(s), img_i, np.asarray(b, dtype=np.float32)))
    rows.sort(key=lambda r: r[0], reverse=True)
    matched = [np.zeros(len(g), dtype=bool) for g in gt_boxes]
    tp = np.zeros(len(rows), dtype=np.float64)
    fp = np.zeros(len(rows), dtype=np.float64)
    for k, (_score, img_i, box) in enumerate(rows):
        gts = gt_boxes[img_i]
        if len(gts) == 0:
            fp[k] = 1.0
            continue
        ious = box_iou(
            torch.as_tensor(box[None, :]), torch.as_tensor(np.asarray(gts, dtype=np.float32))
        )[0].numpy()
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[img_i][j]:
            tp[k] = 1.0
            matched[img_i][j] = True
        else:
            fp[k] = 1.0
    if len(rows) == 0:
        return 0.0
    tpc = np.cumsum(tp)
    fpc = np.cumsum(fp)
    recalls = tpc / n_gt
    precisions = tpc / np.maximum(tpc + fpc, 1e-12)
    return _ap_101(recalls, precisions)


def compute_detection_map(
    detections: List[dict],
    gt_targets: List[dict],
    num_classes: int,
    iou_thresholds: np.ndarray,
    score_thresh: float,
) -> dict:
    """
    detections/gt_targets: torchvision-style dicts with boxes, labels
    (1..num_classes; 0 = background is ignored). detections also have scores.
    """
    per_class_ap50 = []
    per_class_mean = []
    for c in range(1, num_classes + 1):
        pred_boxes, pred_scores, gt_boxes = [], [], []
        for det, gt in zip(detections, gt_targets):
            labels = det["labels"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            boxes = det["boxes"].detach().cpu().numpy()
            keep = (labels == c) & (scores >= score_thresh)
            pred_boxes.append(boxes[keep])
            pred_scores.append(scores[keep])
            gl = gt["labels"].detach().cpu().numpy()
            gb = gt["boxes"].detach().cpu().numpy()
            gt_boxes.append(gb[gl == c])
        aps = [
            _class_ap_at_iou(pred_boxes, pred_scores, gt_boxes, float(t)) for t in iou_thresholds
        ]
        aps = [a for a in aps if not np.isnan(a)]
        if not aps:
            continue
        per_class_ap50.append(aps[0])
        per_class_mean.append(float(np.mean(aps)))
    if not per_class_ap50:
        return {"map50": 0.0, "map5095": 0.0, "per_class_ap50": []}
    return {
        "map50": float(np.mean(per_class_ap50)),
        "map5095": float(np.mean(per_class_mean)),
        "per_class_ap50": [float(x) for x in per_class_ap50],
    }


def print_detection_map_table(metrics: dict, iou_thresholds: np.ndarray, split_label: str) -> None:
    thr_list = ", ".join(f"{t:.2f}" for t in iou_thresholds)
    print("\n" + "=" * 72)
    print(f" FASTER R-CNN DETECTION (RPN + RoI) — {split_label}")
    print("=" * 72)
    print(
        "Ground truth: one whole-image box per sample (weak supervision from ImageFolder).\n"
        "A predicted box is a true positive if IoU with that image box >= threshold "
        "and the class matches.\n"
    )
    print(f"{'Metric':<28} {'Value':>12}")
    print(f"{'-' * 28} {'-' * 12}")
    print(f"{'mAP@0.5':<28} {metrics['map50']:>12.4f}")
    print(f"{'mAP@[0.5:0.95]':<28} {metrics['map5095']:>12.4f}")
    print(f"{'IoU threshold(s) evaluated':<28} {thr_list}")
    print()


@torch.no_grad()
def evaluate_fasterrcnn_map(
    model: torch.nn.Module,
    dataset,
    idx_map: np.ndarray,
    device: torch.device,
    *,
    score_thresh: float,
    iou_thresholds: np.ndarray,
    split_label: str,
    num_classes: int,
) -> Optional[dict]:
    encoder = getattr(model, "fpn_encoder", None)
    if encoder is None or not hasattr(model, "detect"):
        return None
    if getattr(model, "_rpn_roi_from_coco", False):
        print(
            f"[{split_label}] RPN/RoI heads are COCO-initialized (not yet trained on "
            "rock art). mAP below is a baseline; run python train_rockart_model.py "
            "to train the detector."
        )
    encoder.detector.roi_heads.score_thresh = float(score_thresh)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    detections: List[dict] = []
    gts: List[dict] = []

    model.eval()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        mapped = int(idx_map[int(labels[0].item())])
        det = model.detect(imgs)[0]
        h, w = int(imgs.shape[-2]), int(imgs.shape[-1])
        gt = image_level_targets(torch.tensor([mapped], device=device), h, w, device)[0]
        detections.append(det)
        gts.append(gt)
    metrics = compute_detection_map(
        detections, gts, num_classes, iou_thresholds, score_thresh
    )
    print_detection_map_table(metrics, iou_thresholds, split_label)
    return metrics


def evaluate_metrics(
    model: Optional[torch.nn.Module] = None,
    class_names: Optional[List[str]] = None,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    device: Optional[torch.device] = None,
    *,
    with_ttest: Optional[bool] = None,
    dataset_root: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Run classification evaluation on ImageFolder splits under dataset_root.

    Labels are remapped from folder class names onto checkpoint class_names.
    Returns split name -> accuracy (None if skipped).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if with_ttest is None:
        with_ttest = bool(cfg.eval.with_ttest)
    n_bootstrap = int(cfg.eval.n_bootstrap)
    alpha = float(cfg.eval.alpha)
    batch_size = int(cfg.eval.batch_size)
    score_thresh = float(cfg.eval.score_thresh)
    iou_start = float(cfg.eval.iou_start)
    iou_end = float(cfg.eval.iou_end)
    iou_step = float(cfg.eval.iou_step)
    iou_thresholds = np.arange(iou_start, iou_end + 1e-9, iou_step)
    if model is None or class_names is None:
        model, class_names = load_petroglyphnet(device=device)
    else:
        model = model.to(device)
        model.eval()

    print("\n" + "=" * 72)
    print(" EVALUATION METRICS (petroglyphnet.pth)")
    print("=" * 72)

    accuracies: Dict[str, Optional[float]] = {}
    split_predictions: Dict[str, dict] = {}

    for split in splits:
        root = str(split_dir(split, cfg, dataset_root=dataset_root))
        print("\n" + "#" * 72)
        print(f" DATASET: {split.upper()}  ->  {root}")
        print("#" * 72)
        result = _eval_one_split(
            model, class_names, root, split, device, n_bootstrap, alpha, batch_size
        )
        if result is None:
            accuracies[split] = None
        else:
            accuracies[split] = result["accuracy"]
            split_predictions[split] = result
            evaluate_fasterrcnn_map(
                model,
                result["dataset"],
                result["idx_map"],
                device,
                score_thresh=score_thresh,
                iou_thresholds=iou_thresholds,
                split_label=split,
                num_classes=len(class_names),
            )

    print_examiner_validation_overfitting_report(accuracies)
    if with_ttest and split_predictions:
        print_ttest_report(split_predictions, n_classes=len(class_names), alpha=alpha)
    return accuracies


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PetroglyphFusionNet")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ttest", action="store_true", help="Also print appendix t-tests")
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    evaluate_metrics(
        splits=splits,
        with_ttest=args.ttest,
        dataset_root=args.dataset,
    )


if __name__ == "__main__":
    main()
