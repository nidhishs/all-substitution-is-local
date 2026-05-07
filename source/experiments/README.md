# Experiments

All commands are run from `source/`:

```bash
cd source/
```

---

## Data Pipeline

### Step 1 — DenseNet inference

Six DenseNet-121 checkpoints are evaluated on the 500 CheXpert test images. Inference outputs a competition-format CSV and `inference.log` to `data/prepared/chexpert/predictions/<model>/` (one directory per model).

```bash
PYTHONPATH=. uv run python -m data.chexpert infer --weights densenet121-res224-all
PYTHONPATH=. uv run python -m data.chexpert infer --weights densenet121-res224-nih
PYTHONPATH=. uv run python -m data.chexpert infer --weights densenet121-res224-pc
PYTHONPATH=. uv run python -m data.chexpert infer --weights densenet121-res224-chex
PYTHONPATH=. uv run python -m data.chexpert infer --weights densenet121-res224-mimic_nb
PYTHONPATH=. uv run python -m data.chexpert infer --weights densenet121-res224-mimic_ch
```

> `densenet121-res224-rsna` is excluded — its pathology set does not include the 5 CheXpert conditions.

### Step 2 — Prepare pair artefacts

Each inference CSV is prepared into pair artefacts under `data/prepared/chexpert/<subset>/<model>/pairs/`. The `--readers` flag controls which reader subset is prepared:

- `--readers gt` — 5 GT readers × 5 conditions = **25 pairs** (main paper)
- `--readers all` — 8 readers × 5 conditions = **40 pairs** (appendix robustness)
- `--readers bm` — 3 benchmark readers × 5 conditions = 15 pairs

For the **main paper** (§4.2, §4.3), prepare only `densenet121-res224-chex` with GT readers:

```bash
PYTHONPATH=. uv run python -m data.chexpert prepare \
  --predictions data/prepared/chexpert/predictions/densenet121-res224-chex/densenet121-res224-chex.csv \
  --readers gt
```

For the **appendix robustness sweep** (all 6 DenseNet checkpoints, all readers):

```bash
for weights in densenet121-res224-all densenet121-res224-nih densenet121-res224-pc \
               densenet121-res224-chex densenet121-res224-mimic_nb densenet121-res224-mimic_ch; do
  PYTHONPATH=. uv run python -m data.chexpert prepare \
    --predictions "data/prepared/chexpert/predictions/${weights}/${weights}.csv" \
    --readers all
done
```

Each prepare command writes pair artefacts to `data/prepared/chexpert/<readers>/<model>/pairs/`.

---

## Experiment 1 — Synthetic Estimator Validation

```bash
PYTHONPATH=. uv run python -m experiments experiment-1
```

Runs a synthetic grid over three conditions (C1, C2, C3) and six (G, K) configurations at N=10,000 samples with seed=0. Outputs `results.json` and `summary.json` to `results/experiment_1/run_<hex>/`.

---

## Experiment 2 — BR̂ Dissociation on Real Reader Data

### Main paper (§4.2) — 25 pairs

`densenet121-res224-chex` × 5 GT readers × 5 conditions. Requires `--readers gt` prepare step above.

```bash
PYTHONPATH=. uv run python -m experiments experiment-2 \
  --dataset chexpert/gt --model densenet121-res224-chex
```

Outputs `results.json` to `results/experiment_2/run_<hex>/`.

### Appendix robustness — 240 pairs

All 6 DenseNet checkpoints × 8 readers × 5 conditions. Requires `--readers all` prepare steps above.

```bash
PYTHONPATH=. uv run python -m experiments experiment-2 --dataset chexpert/all
```

---

## Experiment 3 — Review Allocation

### Synthetic regime

```bash
PYTHONPATH=. uv run python -m experiments experiment-3 synthetic
```

Evaluates allocation policies (BR_hat, Margin, Entropy, Residual, L2D, Random, Oracle) on 3 synthetic pair configurations at N=10,000, seed=0. Outputs `results.json` to `results/experiment_3/run_<hex>/`.

### Real regime — main paper (§4.3) — 25 pairs

Same 25 pairs as Experiment 2 main. Requires `--readers gt` prepare step above.

```bash
PYTHONPATH=. uv run python -m experiments experiment-3 real \
  --dataset chexpert/gt --model densenet121-res224-chex
```

### Real regime — appendix — 240 pairs

```bash
PYTHONPATH=. uv run python -m experiments experiment-3 real --dataset chexpert/all
```

Outputs `results.json` to `results/experiment_3/run_<hex>/`.
