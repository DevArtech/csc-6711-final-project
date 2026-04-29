#!/bin/bash
#SBATCH --job-name=recsys_bayesian_mf
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=recsys/runs/bayesian_mf/recsys_bayesian_mf_%j.out
#SBATCH --error=recsys/runs/bayesian_mf/recsys_bayesian_mf_%j.err

set -euo pipefail

ENV=${ENV:-/data/csc4611/conda-csc4611}
PYTHON_BIN="$ENV/bin/python"
OUTPUT_DIR=${OUTPUT_DIR:-recsys/runs/bayesian_mf}
DATA_DIR=${DATA_DIR:-recsys/data/processed}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p "$OUTPUT_DIR"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" -u recsys/train_bayesian_mf.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
