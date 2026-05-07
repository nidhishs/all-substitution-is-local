# Data Pipeline

Two datasets are supported:
- **CheXpert** — main empirical study (§4.2, §4.3 + appendix robustness).
- **CIFAR-10H** — theory-confirming negative control (Appendix B.2).

All commands are run from `source/`.

## Raw data

Place under `data/raw/`:

### CheXpert
- `CheXpert/test/` — CheXpert v1.0 test images (500 studies). Download from https://stanfordaimi.azurewebsites.net/datasets/23c56a08-b312-4539-b7f8-3067d5114a58.
- `cheXpert-test-set-labels/` — Radiologist ground-truth CSVs. Clone https://github.com/stanford-aimi/CheXpert-test-set-labels into this directory.

### CIFAR-10H
- `cifar10h/cifar-10-batches-py/` — Standard CIFAR-10 test batch (`test_batch` pickle).
- `cifar10h/cifar10h-raw.csv` — Per-annotator CIFAR-10H labels. Download from https://github.com/jcpeterson/cifar-10h.

## CheXpert pipeline

**Step 1 — DenseNet inference** (writes `data/prepared/chexpert/predictions/<model>/<model>.csv`):

```bash
PYTHONPATH=. uv run python -m data.chexpert infer --weights densenet121-res224-chex
# options: --weights TEXT  --batch-size INT  --force
```

**Step 2 — Prepare pair artefacts** (writes `data/prepared/chexpert/<readers>/<model>/pairs/*.npz`):

```bash
PYTHONPATH=. uv run python -m data.chexpert prepare \
  --predictions data/prepared/chexpert/predictions/densenet121-res224-chex/densenet121-res224-chex.csv \
  --readers gt
# options: --readers [gt|bm|all]  --limit INT
```

Reader subsets:
- `gt` — 5 GT readers × 5 conditions = 25 pairs (main paper).
- `bm` — 3 benchmark readers × 5 conditions = 15 pairs.
- `all` — 8 readers × 5 conditions = 40 pairs (appendix robustness).

## CIFAR-10H pipeline

**Step 1 — Pretrained classifier inference** (writes `data/prepared/cifar10h/<weights>/<weights>.csv`):

```bash
PYTHONPATH=. uv run python -m data.cifar10h infer --weights cifar10_resnet20
# options: --weights TEXT  --batch-size INT  --force
```

Supported weights (from `chenyaofo/pytorch-cifar-models`): `cifar10_resnet20`, `cifar10_resnet32`, `cifar10_resnet44`, `cifar10_resnet56`, `cifar10_vgg16_bn`, `cifar10_mobilenetv2_x1_0`, `cifar10_repvgg_a2`.

**Step 2 — Prepare pair artefacts** (writes `data/prepared/cifar10h/<model>/pairs/*.npz`, 10 classes × top-50 annotators = 500 pairs per model; pairs with fewer than 10 positives or 10 negatives are skipped):

```bash
PYTHONPATH=. uv run python -m data.cifar10h prepare \
  --predictions data/prepared/cifar10h/cifar10_resnet20/cifar10_resnet20.csv
# options: --limit INT
```

## Validate

Validate every pair artefact under `data/prepared/`:

```bash
PYTHONPATH=. uv run python -m data validate
```
