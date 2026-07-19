# TriAD-PCB

**TriAD-PCB: Online Few-Shot Defect Detection and Segmentation for Printed Circuit Boards**.

The project follows the paper’s three-branch design:

- **Text Branch**: semantic anchors, normal/abnormal prompts, and prompt-conditioned feature generation
- **Online Branch**: causal historical memory with redundancy-aware sampling and anomaly-aware gating
- **Reference Branch**: static few-shot support bank for prototype matching
- **Fusion**: confidence-aware weighting, residual refinement, and top-q pooling

## Requirements

- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `pyyaml`

Install them with:

```bash
pip install -r requirements.txt
```

## Dataset format

```text
DATA_ROOT/
├─ images/
│  ├─ 0001.png
│  ├─ 0002.png
│  └─ ...
├─ masks/                # optional
│  ├─ 0001.png
│  ├─ 0002.png
│  └─ ...
└─ labels.csv             # optional
```

## Training

```bash
python scripts/train.py \
  --data-root /path/to/train_dataset \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-dir checkpoints
```

## Streaming inference

```bash
python scripts/test_stream.py \
  --reference-root /path/to/reference_dataset \
  --stream-root /path/to/stream_dataset \
  --checkpoint checkpoints/epoch_50.pt \
  --out outputs/predictions.json
```
