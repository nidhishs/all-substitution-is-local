#!/usr/bin/env bash
# Submit Experiment 3 to the Ray cluster at 192.168.1.163.
#
# Prerequisites (run locally first):
#   .venv/bin/python extract_chexpert_arrays.py
#
# Usage:
#   bash source/submit_experiment3.sh              # CheXpert + CIFAR-10H
#   bash source/submit_experiment3.sh --skip-cifar10h  # CheXpert only
#   bash source/submit_experiment3.sh --demo           # synthetic pipeline test
#
# Resource allocation:
#   --entrypoint-num-cpus 2   driver is orchestration-only (ray.init / ray.get / JSON)
#   --entrypoint-num-gpus 0   GPU is claimed solely by prepare_cifar10h_data task
#   Per allocation task:      num_cpus=2  (sklearn logistic regression, OMP limited)
#   CIFAR-10H prep task:      num_gpus=1, num_cpus=4  (ResNet-18 inference)
#   Peak cluster load:        1 GPU  +  ~14 CPUs
#
# Results are written to ${CLUSTER_RESULTS}/ on the cluster head node and
# pulled back to source/results/ via scp.

set -euo pipefail

CLUSTER_ADDR="http://192.168.1.163:8265"
CLUSTER_HOST="192.168.1.163"
CLUSTER_USER="${CLUSTER_USER:-${USER}}"   # override: CLUSTER_USER=foo bash submit_...sh
CLUSTER_RESULTS="/tmp/exp3_results"
WORKING_DIR="$(cd "$(dirname "$0")" && pwd)"  # absolute path to source/
LOCAL_RESULTS="${WORKING_DIR}/results"

# Pass through flags like --skip-cifar10h or --demo unchanged
EXTRA_ARGS="${*:-}"

echo "=== Experiment 3: Ray Job Submission ==="
echo "Cluster:       ${CLUSTER_ADDR}"
echo "Working dir:   ${WORKING_DIR}"
echo "Cluster user:  ${CLUSTER_USER}  (override with CLUSTER_USER=...)"
echo "Extra flags:   ${EXTRA_ARGS:-none}"
echo ""

# ── Step 1: Create chexpert_pilot.npz from results.csv ───────────────────────
CHEXPERT_NPZ="${WORKING_DIR}/data/chexpert_pilot.npz"
if [[ ! -f "${CHEXPERT_NPZ}" ]]; then
    echo "[1/3] Extracting CheXpert arrays ..."
    "${WORKING_DIR}/.venv/bin/python" "${WORKING_DIR}/extract_chexpert_arrays.py"
else
    echo "[1/3] ${CHEXPERT_NPZ} already exists ($(du -sh "${CHEXPERT_NPZ}" | cut -f1))."
fi

# NOTE: results.csv currently contains SYNTHETIC pilot data (MCE=0.000).
# For publishable CheXpert results, first run experiment2 with real images:
#   .venv/bin/python experiment2_pilot.py --data-dir data/CheXpert
# Then re-run this script so chexpert_pilot.npz contains real DenseNet beliefs.

# ── Step 2: Submit Ray job ───────────────────────────────────────────────────
# Exclude large data subdirs and the local venv from the working-dir upload.
# Only uploads Python source files + data/chexpert_pilot.npz (~40 KB).
RUNTIME_ENV='{
  "excludes": [
    "data/CheXpert",
    "data/cifar10",
    "data/cheXpert-test-set-labels",
    ".venv",
    "__pycache__",
    "results"
  ]
}'

echo ""
echo "[2/3] Submitting Ray job (working-dir upload excludes large data dirs) ..."
echo "      This will stream output until the job completes."
echo ""

ray job submit \
    --address "${CLUSTER_ADDR}" \
    --working-dir "${WORKING_DIR}" \
    --runtime-env-json "${RUNTIME_ENV}" \
    --entrypoint-num-cpus 2 \
    --entrypoint-num-gpus 0 \
    --entrypoint-memory 4000000000 \
    -- python experiment3_ray.py \
        --chexpert-data data/chexpert_pilot.npz \
        --rewards R1 R2 R3 \
        --budgets 0.05 0.10 0.20 0.50 \
        --bootstrap 1000 \
        --output-dir "${CLUSTER_RESULTS}" \
        ${EXTRA_ARGS}

echo ""
echo "[3/3] Pulling results from ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_RESULTS}/ ..."
mkdir -p "${LOCAL_RESULTS}"

scp "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_RESULTS}/experiment3_chexpert_pilot.json" \
    "${LOCAL_RESULTS}/" 2>/dev/null \
    && echo "  Pulled experiment3_chexpert_pilot.json" \
    || echo "  Warning: chexpert result not found on cluster"

scp "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_RESULTS}/experiment3_cifar10h.json" \
    "${LOCAL_RESULTS}/" 2>/dev/null \
    && echo "  Pulled experiment3_cifar10h.json" \
    || echo "  Warning: CIFAR-10H result not found (expected if --skip-cifar10h was set)"

scp "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_RESULTS}/experiment3_demo.json" \
    "${LOCAL_RESULTS}/" 2>/dev/null \
    && echo "  Pulled experiment3_demo.json" \
    || true  # only present with --demo

echo ""
echo "=== Done ==="
ls -lh "${LOCAL_RESULTS}"/experiment3_*.json 2>/dev/null || echo "No experiment3 JSON files found locally."
