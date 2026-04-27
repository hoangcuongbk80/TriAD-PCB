# TriAD-PCB

A runnable PyTorch reference implementation of **TriAD-PCB: Online Few-Shot Defect Detection and Segmentation for Printed Circuit Boards**.

The project follows the paper’s three-branch design:

- **Text Branch**: semantic anchors, normal/abnormal prompts, and prompt-conditioned feature generation
- **Online Branch**: causal historical memory with redundancy-aware sampling and anomaly-aware gating
- **Reference Branch**: static few-shot support bank for prototype matching
- **Fusion**: confidence-aware weighting, residual refinement, and top-q pooling

## Project layout

```text
triad_pcb/
├─ configs/triad_pcb.yaml
├─ scripts/
│  ├─ train.py
│  ├─ test_stream.py
│  └─ export_metrics.py
├─ src/
│  ├─ models/
│  ├─ prompts/
│  ├─ memory/
│  ├─ losses/
│  ├─ data/
│  ├─ eval/
│  └─ utils/
└─ requirements.txt
```

## Requirements

The code is written for Python 3.9+ and PyTorch.

Typical dependencies include:

- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `pyyaml`

Install them with:

```bash
pip install -r requirements.txt
```

## Expected dataset format

The provided loaders expect a simple folder structure:

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

### `labels.csv`

If present, it should contain columns:

```csv
filename,label
0001.png,0
0002.png,1
```

Where `label = 0` means normal and `label = 1` means anomalous.

If `masks/` exists, each mask should use the same filename as the image and contain a binary anomaly map.

## Training

Run offline training with:

```bash
python scripts/train.py \
  --data-root /path/to/train_dataset \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-dir checkpoints
```

The training script:

- loads images from `images/`
- uses masks when available
- optimizes the full TriAD-PCB model
- saves checkpoints after each epoch

## Streaming inference

Run causal test-time inference with a reference set and a stream set:

```bash
python scripts/test_stream.py \
  --reference-root /path/to/reference_dataset \
  --stream-root /path/to/stream_dataset \
  --checkpoint checkpoints/epoch_50.pt \
  --out outputs/predictions.json
```

This script:

- builds the reference bank from the reference images
- resets the online memory
- processes test images one by one in order
- stores image-level anomaly scores in JSON format

## Metrics

Compute image-level metrics from predictions and labels:

```bash
python scripts/export_metrics.py \
  --pred-json outputs/predictions.json \
  --labels-json /path/to/labels.json \
  --out outputs/metrics.json
```

The metrics helper currently reports:

- `I-AUC`
- `I-AP`
- `I-F1-max`

## Configuration

The default hyperparameters are stored in:

```text
configs/triad_pcb.yaml
```

You can adjust:

- embedding dimension
- patch size
- number of anchors
- online memory capacity
- sampling ratio
- fusion temperature
- residual refinement weight

## Notes

- The backbone is implemented as a lightweight CLIP-like PyTorch module so the project can run without external model downloads.
- The code is structured so you can later replace that backbone with a real CLIP/OpenCLIP implementation.
- The current implementation is a practical reference base for experimentation and extension.
