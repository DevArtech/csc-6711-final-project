# Recommender Performance Runbook

This document gives a practical plan to:

1. get stronger evidence that Bayesian MF is better before expensive training, and  
2. run a full-scale experiment that fits in about 24 hours.

It is written for this repository's existing scripts and SLURM setup.

## Goal and decision rule

Primary goal: show `bayesian_mf` beats both `static_mf` and `sequential` in a consistent way.

Use this promotion rule before full-scale:

- Bayesian wins on average across seeds for `ndcg@10` and `hr@10`.
- Bayesian is not worse than the best baseline by more than 1-2% on `rmse`.
- On drift subset (`drift_subset_summary.json`), Bayesian has a clear ranking gain (`ndcg@10` preferred).

If these are not met in medium-scale runs, do not spend full budget yet.

## Phase 0: one-time setup

From repo root:

```bash
python recsys/data/download_ml1m.py
```

Build processed data (full):

```bash
python recsys/data/preprocess.py \
  --input-dir recsys/data/raw/ml-1m \
  --output-dir recsys/data/processed \
  --min-interactions 20
```

Create drift stream metadata (recommended):

```bash
python recsys/data/drift_simulator.py \
  --interactions-csv recsys/data/processed/interactions.csv \
  --output-csv recsys/data/processed/interactions_drift.csv \
  --output-meta recsys/data/processed/drift_meta.json \
  --num-users 1000 \
  --seed 42
```

## Phase 1: cheap high-signal checks (no retraining)

Use existing checkpoints and reduce metric variance by evaluating multiple seeds with more negatives.

```bash
for s in 0 1 2 3 4; do
  python recsys/compare_recommenders.py \
    --data-dir recsys/data/processed \
    --static-checkpoint recsys/runs/static_mf/static_mf.pt \
    --sequential-checkpoint recsys/runs/sequential/sequential.pt \
    --bayesian-checkpoint recsys/runs/bayesian_mf/bayesian_mf.pt \
    --n-negatives 500 \
    --seed "$s" \
    --output-dir "recsys/runs/compare_seed_${s}"
done
```

Why this matters:

- `compare_recommenders.py` uses random negative sampling; one seed can be noisy.
- More negatives (`--n-negatives 500`) makes ranking metrics more stable.

If Bayesian is not competitive here, tune first before scaling up.

## Phase 2: medium-scale training for model selection

Run on a subset of users to keep cost low but preserve realistic behavior.

### 2.1 Build medium dataset

```bash
python recsys/data/preprocess.py \
  --input-dir recsys/data/raw/ml-1m \
  --output-dir recsys/data/processed_medium \
  --min-interactions 20 \
  --max-users 1000
```

```bash
python recsys/data/drift_simulator.py \
  --interactions-csv recsys/data/processed_medium/interactions.csv \
  --output-csv recsys/data/processed_medium/interactions_drift.csv \
  --output-meta recsys/data/processed_medium/drift_meta.json \
  --num-users 300 \
  --seed 42
```

### 2.2 Train one baseline set (fixed)

```bash
python recsys/train_static_mf.py \
  --data-dir recsys/data/processed_medium \
  --output-dir recsys/runs_medium/static_mf \
  --epochs 8
```

```bash
python recsys/train_sequential.py \
  --data-dir recsys/data/processed_medium \
  --output-dir recsys/runs_medium/sequential \
  --epochs 6
```

### 2.3 Bayesian sweep (small grid)

Run 3-4 Bayesian configs and keep the same baselines:

```bash
python recsys/train_bayesian_mf.py \
  --data-dir recsys/data/processed_medium \
  --output-dir recsys/runs_medium/bayes_f100_n050 \
  --warm-epochs 8 \
  --forgetting 1.0 \
  --noise-var 0.5 \
  --prior-var 1.0
```

```bash
python recsys/train_bayesian_mf.py \
  --data-dir recsys/data/processed_medium \
  --output-dir recsys/runs_medium/bayes_f098_n050 \
  --warm-epochs 8 \
  --forgetting 0.98 \
  --noise-var 0.5 \
  --prior-var 1.0
```

```bash
python recsys/train_bayesian_mf.py \
  --data-dir recsys/data/processed_medium \
  --output-dir recsys/runs_medium/bayes_f095_n075 \
  --warm-epochs 8 \
  --forgetting 0.95 \
  --noise-var 0.75 \
  --prior-var 1.0
```

### 2.4 Evaluate each Bayesian config across seeds

Example for one Bayesian checkpoint:

```bash
for s in 0 1 2; do
  python recsys/compare_recommenders.py \
    --data-dir recsys/data/processed_medium \
    --drift-meta recsys/data/processed_medium/drift_meta.json \
    --static-checkpoint recsys/runs_medium/static_mf/static_mf.pt \
    --sequential-checkpoint recsys/runs_medium/sequential/sequential.pt \
    --bayesian-checkpoint recsys/runs_medium/bayes_f098_n050/bayesian_mf.pt \
    --n-negatives 300 \
    --seed "$s" \
    --output-dir "recsys/runs_medium/compare_bayes_f098_n050_seed_${s}"
done
```

Pick the Bayesian config that is most consistently best on ranking and competitive on rating.

## Phase 3: full-scale run designed for ~24 hours

Use the winning Bayesian config from Phase 2.

### 3.1 Recommended scheduling strategy

- Submit all three trainings in parallel (already done by `submit_recsys_pipeline.sh`).
- Keep one compare job dependency-gated after training.
- Use one main seed for the expensive full run.
- Optional: run 1 additional compare-only seed if time remains.

### 3.2 Full-scale pipeline command

```bash
PARTITION=teaching \
ENV=/data/csc4611/conda-csc4611 \
DATA_DIR=recsys/data/processed \
AUTO_PREPROCESS=1 \
MIN_INTERACTIONS=20 \
STATIC_OUT=recsys/runs/full_static_mf \
SEQUENTIAL_OUT=recsys/runs/full_sequential \
BAYESIAN_OUT=recsys/runs/full_bayesian_mf \
COMPARE_OUT=recsys/runs/full_compare \
bash recsys/submit_recsys_pipeline.sh
```

If you want drift-focused comparison, also set:

```bash
DRIFT_META=recsys/data/processed/drift_meta.json
```

### 3.3 Passing Bayesian hyperparameters through SLURM

`submit_recsys_pipeline.sh` forwards extra arguments to each sbatch script.  
For a targeted full run, submit train jobs manually so only Bayesian gets custom knobs:

```bash
sbatch --partition=teaching \
  --export=ALL,ENV=/data/csc4611/conda-csc4611,DATA_DIR=recsys/data/processed,OUTPUT_DIR=recsys/runs/full_static_mf \
  recsys/recsys_static_mf.sbatch
```

```bash
sbatch --partition=teaching \
  --export=ALL,ENV=/data/csc4611/conda-csc4611,DATA_DIR=recsys/data/processed,OUTPUT_DIR=recsys/runs/full_sequential \
  recsys/recsys_sequential.sbatch
```

```bash
sbatch --partition=teaching \
  --export=ALL,ENV=/data/csc4611/conda-csc4611,DATA_DIR=recsys/data/processed,OUTPUT_DIR=recsys/runs/full_bayesian_mf \
  recsys/recsys_bayesian_mf.sbatch \
  --warm-epochs 10 \
  --forgetting 0.98 \
  --noise-var 0.5 \
  --prior-var 1.0
```

After all checkpoints exist, run compare:

```bash
sbatch --partition=teaching \
  --export=ALL,ENV=/data/csc4611/conda-csc4611,DATA_DIR=recsys/data/processed,OUTPUT_DIR=recsys/runs/full_compare,STATIC_CKPT=recsys/runs/full_static_mf/static_mf.pt,SEQUENTIAL_CKPT=recsys/runs/full_sequential/sequential.pt,BAYESIAN_CKPT=recsys/runs/full_bayesian_mf/bayesian_mf.pt \
  recsys/recsys_compare.sbatch \
  --drift-meta recsys/data/processed/drift_meta.json \
  --n-negatives 200 \
  --seed 0
```

## 24-hour execution checklist

- Hour 0-1:
  - confirm data exists (`interactions.csv`, `metadata.json`, optional `drift_meta.json`)
  - submit jobs
- Hour 1-18:
  - monitor with `squeue` and `.out/.err` logs
  - verify checkpoints are written
- Hour 18-22:
  - run/finish compare job
  - collect `summary.json`, `drift_subset_summary.json`, `streaming_curves.json`
- Hour 22-24:
  - make decision using the rule at top of this doc
  - if Bayesian wins narrowly, schedule one follow-up compare seed only

## What to report after runs

From compare outputs:

- `summary.json`: overall ranking + rating leaderboard
- `drift_subset_summary.json`: adaptation under drift
- `streaming_curves.json`: whether Bayesian improves over stream updates
- `per_user_results.csv`: error distribution and where Bayesian helps/hurts

Use these to answer: "Is Bayesian consistently better, or only tied/noisy?"
