# Rockart Model v7 (PetroglyphFusionNet)

Classifies Indian rock-art figures into **Buffalo, Deer, Elephant, Horse, Man** by fusing ImageNet ResNet-50 with Faster R-CNN (ResNet-50-FPN + RPN + RoI heads). Classification uses FPN pooled features by default so existing `petroglyphnet.pth` files load. Detection is reported as mAP@0.5 and mAP@[0.5:0.95] with a whole-image box as weak ground truth.

## Setup

```text
pip install -r requirements.txt
```

Edit `config.yaml` (or set `ROCKART_DATASET_ROOT` / `ROCKART_WEIGHTS`) so paths match your machine.

```text
Datasets/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
```

## Train

```text
python train_rockart_model.py
```

Ablation:

```text
python train_rockart_model.py --branch resnet --weights D:/Sunayana/PHD/Project/CNN/Model/resnet_only.pth
python train_rockart_model.py --branch fpn    --weights D:/Sunayana/PHD/Project/CNN/Model/fpn_only.pth
python train_rockart_model.py --branch fusion --fusion concat
python train_rockart_model.py --branch fusion --fusion gated
```

Best weights go to `paths.weights`; last epoch to `petroglyphnet_last.pth`. Log: `outputs/train_log.csv`.

Older FPN-backbone checkpoints load automatically (RPN/RoI start from COCO). For rock-art mAP, retrain. To classify from RoI features instead of FPN GAP, set `model.embed_source: roi` (needs a new training run).

## Predict and evaluate

```text
python app.py
python evaluate_metrics.py
python evaluate_metrics.py --ttest
```

Unlabeled site photos:

```text
python app.py --unlabeled-dir "D:/Sunayana/PHD/Project/CNN/Datasets/Bhimbetka_Sarumaru_talpura" --skip-eval
```

Predictions: `outputs/predictions.csv`. Confusion matrices: `outputs/metrics/`.

## Grad-CAM

```text
python infer_gradcam.py path\to\image.jpg
```

Writes `outputs/gradcam/<name>_resnet.jpg` and `<name>_fpn.jpg`.

## Notes

Denoise + Lab L-channel equalize is **off** by default. Enable with `train.use_preprocess: true` (must match train and eval). This is histogram equalization on Lab L, not PCA DStretch.

| Key | Meaning |
|-----|---------|
| `model.branch` | `resnet` / `fpn` / `fusion` |
| `model.fusion` | `concat` or `gated` |
| `model.embed_source` | `fpn` (default) or `roi` |
| `train.freeze_epochs` | train head only, then unfreeze backbones |
| `train.accum_steps` | effective batch = `batch_size * accum_steps` |
| `train.class_weight` | `effective` / `balanced` / `inverse` / `none` |
| `eval.with_ttest` | appendix t-tests |
