## Raw data

Place the following under `data/raw/` before running the pipeline:

- `CheXpert/test/` — CheXpert v1.0 test images (500 studies). Download from https://stanfordaimi.azurewebsites.net/datasets/23c56a08-b312-4539-b7f8-3067d5114a58
- `cheXpert-test-set-labels/` — Radiologist ground-truth CSVs. Clone https://github.com/stanford-aimi/CheXpert-test-set-labels into this directory.

## Pipeline

All commands are run from `source/`.

**Step 1 — DenseNet inference** (writes `data/prepared/chexpert/<weights>/<weights>.csv`):

```bash
python -m data.chexpert infer
# options: --weights TEXT  --batch-size INT  --force
```

**Step 2 — Prepare pair artefacts** (writes `data/prepared/chexpert/<model>/pairs/*.npz`):

```bash
python -m data.chexpert prepare \
  --predictions data/prepared/chexpert/densenet121-res224-chex/densenet121-res224-chex.csv
# options: --readers [gt|benchmark|all]  --limit INT
```

**Step 3 — Validate** (checks all artefacts under `data/prepared/`):

```bash
python -m data validate
```
