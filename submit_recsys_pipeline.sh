#!/bin/bash
set -euo pipefail

# Submits static/sequential/bayesian training jobs, then compare after all succeed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PARTITION="${PARTITION:-teaching}"
ENV_PATH="${ENV:-/data/csc4611/conda-csc4611}"
DATA_DIR="${DATA_DIR:-recsys/data/processed}"
RAW_ML1M_DIR="${RAW_ML1M_DIR:-recsys/data/raw/ml-1m}"
AUTO_PREPROCESS="${AUTO_PREPROCESS:-0}"
MIN_INTERACTIONS="${MIN_INTERACTIONS:-20}"

STATIC_OUT="${STATIC_OUT:-recsys/runs/static_mf}"
SEQUENTIAL_OUT="${SEQUENTIAL_OUT:-recsys/runs/sequential}"
BAYESIAN_OUT="${BAYESIAN_OUT:-recsys/runs/bayesian_mf}"
COMPARE_OUT="${COMPARE_OUT:-recsys/runs/compare}"

STATIC_CKPT="${STATIC_CKPT:-$STATIC_OUT/static_mf.pt}"
SEQUENTIAL_CKPT="${SEQUENTIAL_CKPT:-$SEQUENTIAL_OUT/sequential.pt}"
BAYESIAN_CKPT="${BAYESIAN_CKPT:-$BAYESIAN_OUT/bayesian_mf.pt}"

COMPARE_INTERACTIONS_FILE="${COMPARE_INTERACTIONS_FILE:-interactions.csv}"
DRIFT_META="${DRIFT_META:-$DATA_DIR/drift_meta.json}"
INTERACTIONS_CSV="$DATA_DIR/$COMPARE_INTERACTIONS_FILE"
PYTHON_BIN="$ENV_PATH/bin/python"

echo "Submitting recommender pipeline from: $ROOT_DIR"
echo "Partition: $PARTITION"
echo "Data dir: $DATA_DIR"
echo

if [[ "$AUTO_PREPROCESS" == "1" ]]; then
  echo "AUTO_PREPROCESS=1: preparing processed data before job submission."
  if [[ ! -f "$INTERACTIONS_CSV" ]]; then
    "$PYTHON_BIN" recsys/data/preprocess.py \
      --input-dir "$RAW_ML1M_DIR" \
      --output-dir "$DATA_DIR" \
      --min-interactions "$MIN_INTERACTIONS"
  fi
  if [[ ! -f "$DRIFT_META" ]]; then
    "$PYTHON_BIN" recsys/data/drift_simulator.py \
      --interactions-csv "$INTERACTIONS_CSV" \
      --output-csv "$DATA_DIR/interactions_drift.csv" \
      --output-meta "$DRIFT_META"
  fi
fi

if [[ ! -f "$INTERACTIONS_CSV" ]]; then
  echo "ERROR: Missing processed interactions file: $INTERACTIONS_CSV"
  echo "Run preprocessing first, or set AUTO_PREPROCESS=1."
  echo "Example:"
  echo "  $PYTHON_BIN recsys/data/preprocess.py --input-dir $RAW_ML1M_DIR --output-dir $DATA_DIR"
  exit 1
fi

if [[ ! -f "$DATA_DIR/metadata.json" ]]; then
  echo "ERROR: Missing metadata file: $DATA_DIR/metadata.json"
  exit 1
fi

NUM_INTERACTIONS="$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("num_interactions", 0)))' "$DATA_DIR/metadata.json")"
if [[ "$NUM_INTERACTIONS" -le 0 ]]; then
  echo "ERROR: Processed dataset has zero interactions (metadata.num_interactions=0)."
  echo "Check RAW_ML1M_DIR and preprocessing filters (MIN_INTERACTIONS=$MIN_INTERACTIONS)."
  exit 1
fi

if [[ ! -f "$DRIFT_META" ]]; then
  echo "Warning: drift metadata missing at $DRIFT_META (comparison will run without drift subset metrics)."
fi

STATIC_JOB_ID="$(sbatch --parsable \
  --partition="$PARTITION" \
  --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$STATIC_OUT" \
  recsys/recsys_static_mf.sbatch)"

SEQUENTIAL_JOB_ID="$(sbatch --parsable \
  --partition="$PARTITION" \
  --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$SEQUENTIAL_OUT" \
  recsys/recsys_sequential.sbatch)"

BAYESIAN_JOB_ID="$(sbatch --parsable \
  --partition="$PARTITION" \
  --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$BAYESIAN_OUT" \
  recsys/recsys_bayesian_mf.sbatch)"

COMPARE_JOB_ID="$(sbatch --parsable \
  --partition="$PARTITION" \
  --dependency="afterok:${STATIC_JOB_ID}:${SEQUENTIAL_JOB_ID}:${BAYESIAN_JOB_ID}" \
  --export=ALL,ENV="$ENV_PATH",DATA_DIR="$DATA_DIR",OUTPUT_DIR="$COMPARE_OUT",STATIC_CKPT="$STATIC_CKPT",SEQUENTIAL_CKPT="$SEQUENTIAL_CKPT",BAYESIAN_CKPT="$BAYESIAN_CKPT" \
  recsys/recsys_compare.sbatch \
  --interactions-file "$COMPARE_INTERACTIONS_FILE" \
  --drift-meta "$DRIFT_META")"

echo "Submitted jobs:"
echo "  static_mf:   $STATIC_JOB_ID"
echo "  sequential:  $SEQUENTIAL_JOB_ID"
echo "  bayesian_mf: $BAYESIAN_JOB_ID"
echo "  compare:     $COMPARE_JOB_ID (depends on all training jobs)"
echo
echo "Monitor with:"
echo "  squeue -j $STATIC_JOB_ID,$SEQUENTIAL_JOB_ID,$BAYESIAN_JOB_ID,$COMPARE_JOB_ID"
