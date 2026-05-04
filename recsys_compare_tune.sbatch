#!/bin/bash
#SBATCH --job-name=recsys_compare_tune
#SBATCH --partition=dgx
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

ENV=${ENV:-/data/csc4611/conda-csc4611}
PYTHON_BIN="$ENV/bin/python"
DATA_DIR=${DATA_DIR:-recsys/data/processed}
OUTPUT_DIR=${OUTPUT_DIR:-recsys/runs/compare_tune/default}

STATIC_CKPT=${STATIC_CKPT:-recsys/runs/static_mf/static_mf.pt}
SEQUENTIAL_CKPT=${SEQUENTIAL_CKPT:-recsys/runs/sequential/sequential.pt}
BAYESIAN_CKPT=${BAYESIAN_CKPT:-recsys/runs/bayesian_mf/bayesian_mf.pt}
MODELS=${MODELS:-}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p "$OUTPUT_DIR"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Bayesian checkpoint: $BAYESIAN_CKPT"
echo "Models: ${MODELS:-all}"
echo "Output: $OUTPUT_DIR"

MODELS_ARG=""
[[ -n "$MODELS" ]] && MODELS_ARG="--models $MODELS"

"$PYTHON_BIN" -u recsys/compare_recommenders.py \
  --data-dir "$DATA_DIR" \
  --static-checkpoint "$STATIC_CKPT" \
  --sequential-checkpoint "$SEQUENTIAL_CKPT" \
  --bayesian-checkpoint "$BAYESIAN_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  $MODELS_ARG \
  "$@"
