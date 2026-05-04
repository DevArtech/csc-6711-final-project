#!/bin/bash
set -euo pipefail

# Submits Static MF (BPR loss) training + compare job.
# Reuses existing sequential and bayesian checkpoints.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PARTITION="${PARTITION:-dgx}"
ENV_PATH="${ENV:-/data/csc4611/conda-csc4611}"
DATA_DIR="${DATA_DIR:-recsys/data/processed}"
DRIFT_META="${DRIFT_META:-$DATA_DIR/drift_meta.json}"

STATIC_BPR_OUT="recsys/runs/static_mf_bpr"
COMPARE_OUT="recsys/runs/compare_bpr"

STATIC_BPR_CKPT="$STATIC_BPR_OUT/static_mf.pt"
SEQUENTIAL_CKPT="${SEQUENTIAL_CKPT:-recsys/runs/sequential/sequential.pt}"
BAYESIAN_CKPT="${BAYESIAN_CKPT:-recsys/runs/bayesian_mf/bayesian_mf.pt}"

echo "Submitting Static MF (BPR) from: $ROOT_DIR"
echo "Partition: $PARTITION"
echo

TRAIN_ID="$(sbatch --parsable \
  --partition="$PARTITION" \
  --output="$STATIC_BPR_OUT/train_%j.out" \
  --error="$STATIC_BPR_OUT/train_%j.err" \
  --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$STATIC_BPR_OUT" \
  recsys/recsys_static_mf_bpr.sbatch)"

COMPARE_ID="$(sbatch --parsable \
  --partition="$PARTITION" \
  --output="$COMPARE_OUT/compare_%j.out" \
  --error="$COMPARE_OUT/compare_%j.err" \
  --dependency="afterok:${TRAIN_ID}" \
  --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$COMPARE_OUT",STATIC_CKPT="$STATIC_BPR_CKPT",SEQUENTIAL_CKPT="$SEQUENTIAL_CKPT",BAYESIAN_CKPT="$BAYESIAN_CKPT" \
  recsys/recsys_compare_tune.sbatch \
  --drift-meta "$DRIFT_META")"

echo "  train job:   $TRAIN_ID"
echo "  compare job: $COMPARE_ID (runs after training)"
echo
echo "Monitor: squeue --me"
echo "Results: cat $COMPARE_OUT/summary.json"
