#!/bin/bash
set -euo pipefail

# Submits 4 Bayesian MF tuning jobs in parallel, each followed by a compare job.
# Reuses the already-trained static_mf and sequential checkpoints.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PARTITION="${PARTITION:-dgx}"
ENV_PATH="${ENV:-/data/csc4611/conda-csc4611}"
DATA_DIR="${DATA_DIR:-recsys/data/processed}"
DRIFT_META="${DRIFT_META:-$DATA_DIR/drift_meta.json}"

STATIC_CKPT="${STATIC_CKPT:-recsys/runs/static_mf/static_mf.pt}"
SEQUENTIAL_CKPT="${SEQUENTIAL_CKPT:-recsys/runs/sequential/sequential.pt}"

TUNE_BASE="recsys/runs/bayesian_tune"
COMPARE_BASE="recsys/runs/compare_tune"

# Configs: "label forgetting noise_var"
CONFIGS=(
  "f090_nv050 0.90 0.50"
  "f095_nv050 0.95 0.50"
  "f098_nv050 0.98 0.50"
  "f095_nv025 0.95 0.25"
)

echo "Submitting Bayesian tuning jobs from: $ROOT_DIR"
echo "Partition: $PARTITION"
echo "Static checkpoint:     $STATIC_CKPT"
echo "Sequential checkpoint: $SEQUENTIAL_CKPT"
echo

ALL_COMPARE_IDS=()

for cfg in "${CONFIGS[@]}"; do
  read -r LABEL FORGETTING NOISE_VAR <<< "$cfg"

  BAYES_OUT="$TUNE_BASE/$LABEL"
  COMPARE_OUT="$COMPARE_BASE/$LABEL"
  BAYES_CKPT="$BAYES_OUT/bayesian_mf.pt"

  # Training job
  TRAIN_ID="$(sbatch --parsable \
    --partition="$PARTITION" \
    --output="$BAYES_OUT/train_%j.out" \
    --error="$BAYES_OUT/train_%j.err" \
    --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$BAYES_OUT",FORGETTING="$FORGETTING",NOISE_VAR="$NOISE_VAR" \
    recsys/recsys_bayesian_tune.sbatch)"

  # Compare job — waits for training, reuses static+sequential checkpoints
  COMPARE_ID="$(sbatch --parsable \
    --partition="$PARTITION" \
    --output="$COMPARE_OUT/compare_%j.out" \
    --error="$COMPARE_OUT/compare_%j.err" \
    --dependency="afterok:${TRAIN_ID}" \
    --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$COMPARE_OUT",STATIC_CKPT="$STATIC_CKPT",SEQUENTIAL_CKPT="$SEQUENTIAL_CKPT",BAYESIAN_CKPT="$BAYES_CKPT" \
    recsys/recsys_compare_tune.sbatch \
    --drift-meta "$DRIFT_META")"

  ALL_COMPARE_IDS+=("$COMPARE_ID")

  echo "[$LABEL]  forgetting=$FORGETTING  noise_var=$NOISE_VAR"
  echo "  train job:   $TRAIN_ID"
  echo "  compare job: $COMPARE_ID"
  echo
done

echo "Monitor all jobs:"
TRAIN_IDS=$(squeue --me --noheader --format="%i" | head -20 | tr '\n' ',' | sed 's/,$//')
echo "  squeue --me"
echo
echo "Compare job IDs: ${ALL_COMPARE_IDS[*]}"
echo
echo "When done, collect results with:"
echo "  python recsys/summarize_tuning.py"
