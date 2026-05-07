#!/usr/bin/env bash
# Reproduce paper results. Run from source/.
#   bash experiments/run_experiments.sh                # main + appendix
#   MAIN_ONLY=1 bash experiments/run_experiments.sh    # main only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # → source/

DENSENET=(all nih pc chex mimic_nb mimic_ch)
SUBSETS=(gt all)
[[ "${MAIN_ONLY:-0}" == "1" ]] && SUBSETS=(gt)

# Wipe previous runs (preserves inference cache in data/prepared/chexpert/predictions/)
rm -rf data/prepared/chexpert/gt data/prepared/chexpert/all data/prepared/chexpert/bm results

# Step 1 — DenseNet inference + prepare
for w in "${DENSENET[@]}"; do
  WEIGHTS="densenet121-res224-${w}"
  PYTHONPATH=. uv run python -m data.chexpert infer --weights "$WEIGHTS"
  for s in "${SUBSETS[@]}"; do
    PYTHONPATH=. uv run python -m data.chexpert prepare \
      --predictions "data/prepared/chexpert/predictions/${WEIGHTS}/${WEIGHTS}.csv" --readers "$s"
  done
done

# Experiment 1
PYTHONPATH=. uv run python -m experiments experiment-1

# Experiment 2 — main (§4.2)
PYTHONPATH=. uv run python -m experiments experiment-2 --dataset chexpert/gt --model densenet121-res224-chex
# Experiment 2 — appendix
[[ "${MAIN_ONLY:-0}" != "1" ]] && \
  PYTHONPATH=. uv run python -m experiments experiment-2 --dataset chexpert/all

# Experiment 3
PYTHONPATH=. uv run python -m experiments experiment-3 synthetic
PYTHONPATH=. uv run python -m experiments experiment-3 real --dataset chexpert/gt --model densenet121-res224-chex
[[ "${MAIN_ONLY:-0}" != "1" ]] && \
  PYTHONPATH=. uv run python -m experiments experiment-3 real --dataset chexpert/all
