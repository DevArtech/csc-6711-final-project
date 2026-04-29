# Belief-Aware Recommender Experiments

This directory contains a self-contained project for comparing:

- static matrix factorization (MF),
- sequential recommendation with auxiliary heads, and
- Bayesian online MF (belief-aware user preference modeling).

## Quick workflow

1. Download MovieLens-1M:

```bash
python recsys/data/download_ml1m.py
```

2. Build chronological splits:

```bash
python recsys/data/preprocess.py --input-dir recsys/data/raw/ml-1m --output-dir recsys/data/processed
```

3. Train models:

```bash
python recsys/train_static_mf.py --data-dir recsys/data/processed --output-dir recsys/runs/static_mf
python recsys/train_sequential.py --data-dir recsys/data/processed --output-dir recsys/runs/sequential
python recsys/train_bayesian_mf.py --data-dir recsys/data/processed --output-dir recsys/runs/bayesian_mf
```

4. Run head-to-head comparison:

```bash
python recsys/compare_recommenders.py \
  --data-dir recsys/data/processed \
  --static-checkpoint recsys/runs/static_mf/static_mf.pt \
  --sequential-checkpoint recsys/runs/sequential/sequential.pt \
  --bayesian-checkpoint recsys/runs/bayesian_mf/bayesian_mf.pt \
  --output-dir recsys/runs/compare
```

## SLURM Orchestrator

Submit all three training jobs plus dependency-gated comparison:

```bash
bash recsys/submit_recsys_pipeline.sh
```

Common overrides:

```bash
PARTITION=teaching \
ENV=/data/csc4611/conda-csc4611 \
DATA_DIR=recsys/data/processed \
COMPARE_INTERACTIONS_FILE=interactions.csv \
bash recsys/submit_recsys_pipeline.sh
```

If you see `DependencyNeverSatisfied` for the compare job, it means one or more train jobs failed.
The most common cause is missing processed data. You can force local preprocessing first:

```bash
AUTO_PREPROCESS=1 \
RAW_ML1M_DIR=recsys/data/raw/ml-1m \
DATA_DIR=recsys/data/processed \
bash recsys/submit_recsys_pipeline.sh
```

## Notes

- Scripts are intentionally lightweight and reproducible for SLURM workflows.
- The Bayesian model performs online posterior updates during streaming evaluation.
- `recsys/runs/` is ignored by git.
