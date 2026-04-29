#!/bin/bash
#SBATCH --job-name=recsys_compare
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=recsys/runs/compare/recsys_compare_%j.out
#SBATCH --error=recsys/runs/compare/recsys_compare_%j.err

set -euo pipefail

ENV=${ENV:-/data/csc4611/conda-csc4611}
PYTHON_BIN="$ENV/bin/python"
DATA_DIR=${DATA_DIR:-recsys/data/processed}
OUTPUT_DIR=${OUTPUT_DIR:-recsys/runs/compare}

STATIC_CKPT=${STATIC_CKPT:-recsys/runs/static_mf/static_mf.pt}
SEQUENTIAL_CKPT=${SEQUENTIAL_CKPT:-recsys/runs/sequential/sequential.pt}
BAYESIAN_CKPT=${BAYESIAN_CKPT:-recsys/runs/bayesian_mf/bayesian_mf.pt}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p "$OUTPUT_DIR"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "Static checkpoint: $STATIC_CKPT"
echo "Sequential checkpoint: $SEQUENTIAL_CKPT"
echo "Bayesian checkpoint: $BAYESIAN_CKPT"

"$PYTHON_BIN" -u recsys/compare_recommenders.py \
  --data-dir "$DATA_DIR" \
  --static-checkpoint "$STATIC_CKPT" \
  --sequential-checkpoint "$SEQUENTIAL_CKPT" \
  --bayesian-checkpoint "$BAYESIAN_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
