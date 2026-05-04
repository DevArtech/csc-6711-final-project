#!/bin/bash
#SBATCH --job-name=recsys_static_bpr
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

ENV=${ENV:-/data/csc4611/conda-csc4611}
PYTHON_BIN="$ENV/bin/python"
OUTPUT_DIR=${OUTPUT_DIR:-recsys/runs/static_mf_bpr}
DATA_DIR=${DATA_DIR:-recsys/data/processed}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p "$OUTPUT_DIR"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "Loss: bpr  Output: $OUTPUT_DIR"

"$PYTHON_BIN" -u recsys/train_static_mf.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --loss bpr \
  "$@"
