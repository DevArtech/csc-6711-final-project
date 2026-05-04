# CSC-6711 Final Project — Belief-Aware Recommender Systems

## Project Overview
MSOE graduate course (CSC-6711) final project on recommender systems. The core question: **Do "belief-aware" recommenders that continuously update user profiles outperform systems that freeze profiles after training?**

## Research Hypothesis
- Traditional systems (Netflix, Spotify, Amazon) freeze user profiles after training
- Real user preferences drift over time (e.g., horror phase → documentaries phase)
- A system that maintains an evolving belief about each user should recommend better, especially when preferences change
- This is tested via 3 competing models on MovieLens-1M with synthetic preference drift injection

## The Three Models

| Model | File | Key Idea | Hypothesis |
|-------|------|----------|-----------|
| **Static MF** | `static_mf.py` | Classic matrix factorization; learns fixed user/movie fingerprints; never updates | Baseline — expected to perform worst on drift users |
| **Sequential (GRU)** | `sequential_model.py` | Neural RNN that reads full watch history in order; context-aware but doesn't update its learned model | Should outperform Static MF on stable users but lag on drift |
| **Bayesian Online MF** | `bayesian_mf.py` | **Proposed approach** — treats preferences as probability distribution; updates via Bayes' theorem after every interaction; has forgetting factor for recency weighting | Expected to excel on drift users; should match or beat both baselines |

## Dataset: MovieLens-1M
- 1M ratings, 6K users, 4K movies, **with timestamps**
- Split into three stages (per user's timeline):
  - **warm**: Initial training data (all models trained here)
  - **stream**: Interactions replayed chronologically; Bayesian model updates live; models evaluated at checkpoints
  - **test**: Final holdout (never seen during training/streaming)

## The Drift Simulation
- **File**: `drift_simulator.py`
- **Purpose**: Stress-test belief-aware model in the exact scenario it was designed for
- **Method**: Synthetically swap user interaction histories mid-stream for a subset of users
  - Example: User A (Action → Thriller) and User B (Romance → Comedy) swap histories in second half of stream
  - Static MF frozen with old fingerprint; Bayesian model can adapt
- Creates `drift_meta.json` to identify which users experienced drift

## Bayesian Model Details
- Treats user preferences as a **probability distribution**, not a single point estimate
- After each new rating: computes likelihood of the rating given current belief, then updates belief via Bayes' theorem
- **Forgetting factor** (typically 0.95–1.0): old evidence gradually fades so recent behavior matters more
- Mathematically equivalent to a Kalman filter on user preferences (used in GPS, robotics)
- Produces uncertainty estimates (Brier score, ECE) unique to probabilistic output

## Evaluation Metrics
- **HR@10**: Did the true next movie appear in top 10 recommendations?
- **NDCG@10**: Ranking quality (higher reward for true movie near top)
- **RMSE / MAE**: Rating prediction error (lower is better)
- **Brier Score & ECE**: Bayesian-only; measure calibration of predicted uncertainties

## Full Pipeline
```
download_ml1m.py → preprocess.py → [train_static_mf.py, train_sequential.py, train_bayesian_mf.py]
                                    ↓
                        compare_recommenders.py
                        (streaming eval + drift subset eval + plots)
                                    ↓
                        runs/{static_mf,sequential,bayesian_mf}/
                        runs/compare/ → summary.json, drift_subset_summary.json,
                                       streaming_curves.json, per_user_results.csv
```

## Current Status
- **Smoke test** (6 users) completed in `runs_smoke/`:
  - All 3 models roughly tied: HR@10 ≈ 0.33
  - Too small to draw conclusions; need full Rosie run
- **Next step**: Full-scale run on MSOE Rosie supercomputer via SLURM
  - Submit with: `bash submit_recsys_pipeline.sh`
  - All 4 jobs (3 train + 1 compare) submitted with automatic dependency ordering
  - Expected duration: ~24 hours

## Success Criteria (from PERFORMANCE_RUNBOOK.md)
- Bayesian wins on average for NDCG@10 and HR@10
- Bayesian ≤ 1–2% worse on RMSE vs. best baseline
- **On drift subset**: Bayesian shows clear ranking advantage (NDCG@10 preferred)

## Key Files to Know
- `submit_recsys_pipeline.sh`: Submits SLURM jobs with dependencies
- `.sbatch` files: Individual SLURM scripts (static_mf, sequential, bayesian_mf, compare)
- `compare_recommenders.py`: Orchestrates evaluation; produces all metrics
- `streaming_eval.py`: Replays interactions live, checkpointing model performance
- `plot_results.py`: Generates bar charts and streaming curves
- `project.md`: Detailed explanation of problem, models, and pipeline (read for intuition)
- `PERFORMANCE_RUNBOOK.md`: Practical execution guide (3 phases: cheap checks → medium-scale → full-scale)

## Running Locally (Smoke Test)
```bash
python data/download_ml1m.py
python data/preprocess.py --input-dir data/raw/ml-1m --output-dir data/processed
python train_static_mf.py --data-dir data/processed --output-dir runs/static_mf
python train_sequential.py --data-dir data/processed --output-dir runs/sequential
python train_bayesian_mf.py --data-dir data/processed --output-dir runs/bayesian_mf
python compare_recommenders.py \
  --data-dir data/processed \
  --static-checkpoint runs/static_mf/static_mf.pt \
  --sequential-checkpoint runs/sequential/sequential.pt \
  --bayesian-checkpoint runs/bayesian_mf/bayesian_mf.pt \
  --output-dir runs/compare
```

## Running on Rosie (Full Scale)
```bash
PARTITION=teaching \
ENV=/data/csc4611/conda-csc4611 \
DATA_DIR=data/processed \
bash submit_recsys_pipeline.sh
```
