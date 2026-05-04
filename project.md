# CSC-6711 Final Project — Belief-Aware Recommender Systems

> **One-sentence summary:** Can a system that *continuously updates its understanding of you* recommend better movies than one that made up its mind about you during training and never changed?

---

## The Core Problem

Most recommender systems (Netflix, Spotify, Amazon) learn who you are from your past interactions and then **freeze that understanding forever**. In reality, your taste changes — you might go through a horror phase, then switch to documentaries. A system with a frozen snapshot of you from six months ago will keep recommending the wrong things.

```
Traditional system:
  Training ──► [Fixed user profile] ──► Recommendations
               "You like Action"         (never updates)

Belief-aware system:
  Training ──► [Starting belief] ──► Interaction 1 ──► [Updated belief] ──► ...
               "You like Action"    You rate a Drama    "Maybe Drama too?"
```

**Our research question:** Does maintaining a *living, evolving belief* about each user lead to better recommendations — especially when user preferences drift over time?

---

## The Three Models We Compare

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   MODEL 1: Static Matrix Factorization          static_mf.py       │
│   ─────────────────────────────────────────                         │
│   • Classic ML approach                                             │
│   • Learns a fixed "fingerprint" for every user and every movie     │
│   • Recommendation = how well your fingerprint matches the movie's  │
│   • Never updates after training                                     │
│   • The baseline — what we're trying to beat                        │
│                                                                     │
│   MODEL 2: Sequential Recommender              sequential_model.py  │
│   ─────────────────────────────────────────                         │
│   • Uses a GRU (a type of neural network for sequences)             │
│   • Reads your full watch history like a story, in order            │
│   • Predicts: "given everything you've watched, what's next?"       │
│   • Also predicts genre preferences and rating bucket               │
│   • Context-aware but still doesn't update its internal model       │
│                                                                     │
│   MODEL 3: Bayesian Online MF ★              bayesian_mf.py        │
│   ─────────────────────────────────────────                         │
│   • Our proposed approach — the "belief-aware" system               │
│   • Treats user preferences as a probability distribution           │
│     (not just a single fixed value)                                 │
│   • After every new interaction, it revises its belief using        │
│     Bayes' theorem — like updating a hypothesis with new evidence   │
│   • Has a "forgetting factor" so recent behavior matters more       │
│   • The only model that truly adapts in real time                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How the Bayesian Model Thinks

Imagine the model keeps a "confidence cloud" around its guess of your preferences. When it knows little about you, the cloud is wide. As you interact more, the cloud shrinks and shifts.

```
New user                After 5 ratings          After 50 ratings
(high uncertainty)      (less uncertain)          (confident)

    ?????                   ~~~?~~~                  [Action]
  ??     ??               ~~     ~~
??  [???]  ??           ~  [Action] ~              Tight belief.
  ??     ??               ~~     ~~                Confident.
    ?????                   ~~~?~~~
```

The math behind this is **Bayesian inference**: each new rating is evidence that shifts the probability distribution over what kind of movies you like.

The `forgetting` parameter lets old data fade — so if you rated a lot of comedies a year ago but only rate dramas now, the model gradually forgets the comedy signal.

---

## Dataset

**MovieLens-1M** — 1 million movie ratings from 6,000 users across 4,000 movies, with timestamps.

The timestamps are critical: they let us replay interactions in the exact order they happened, simulating a real-world streaming scenario rather than a static train/test split.

```
Timeline of a user's interactions:
─────────────────────────────────────────────────────► time

[──── warm ────][────────── stream ──────────][─ test ─]
  (training)      (model updates happen here)  (final eval)
```

| Split | Purpose |
|---|---|
| **warm** | Initial training data — all models start here |
| **stream** | Interactions replayed one by one — Bayesian model updates live |
| **test** | Final holdout — never seen during training or streaming |

---

## The Drift Experiment

To stress-test the belief-aware model, we synthetically **inject preference drift** for a subset of users.

```
How drift is simulated (drift_simulator.py):

User A watches: Action, Action, Thriller, Action ...
User B watches: Romance, Drama, Romance, Comedy ...

In the second half of the stream, we SWAP their interaction histories:
User A now "watches": Romance, Drama, Romance, Comedy ...
User B now "watches": Action, Action, Thriller, Action ...

This simulates a sudden, hard preference shift.
```

Models are then evaluated separately on these "drift users" — the setting where the Bayesian model should have the biggest advantage, since it can update its belief while Static MF is stuck with its original fingerprint.

---

## How Evaluation Works

Every model gets the same test: given a user, can you find the one movie they actually watched next, hidden among 100 random decoys?

```
Test scenario (evaluate.py):

  Real next movie: [The Matrix]  ← hidden in a pool of 101 candidates
  100 random movies the user hasn't seen

  Model scores all 101 → sorts them → we check where [The Matrix] lands

  If rank 1  → perfect
  If rank 10 → okay
  If rank 50 → bad
```

### Metrics

| Metric | What it measures | Better = |
|---|---|---|
| **HR@K** (Hit Rate) | Did the real movie appear in the top K? | Higher |
| **NDCG@K** | Did it appear near the top of the K? | Higher |
| **Precision@K** | Of top K shown, what fraction were relevant? | Higher |
| **Recall@K** | Of all relevant items, how many did we catch? | Higher |
| **RMSE** | How far off were predicted ratings? | Lower |
| **MAE** | Average absolute rating prediction error | Lower |
| **Brier Score** | Calibration of Bayesian uncertainty (Bayesian model only) | Lower |
| **ECE** | Expected calibration error (Bayesian model only) | Lower |

The Bayesian model gets two extra metrics (Brier + ECE) because it's the only one that produces a full probability distribution over ratings, not just a point estimate.

---

## The Full Pipeline

```
                        ┌─────────────────┐
                        │ MovieLens-1M    │
                        │ (raw data)      │
                        └────────┬────────┘
                                 │
                        download_ml1m.py
                                 │
                        ┌────────▼────────┐
                        │  preprocess.py  │
                        │  (sort by time, │
                        │  split dataset) │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     train_static_mf.py  train_sequential.py  train_bayesian_mf.py
              │                  │                  │
       static_mf.pt       sequential.pt       bayesian_mf.pt
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                      compare_recommenders.py
                      (streaming eval + final eval
                       + drift subset eval + plots)
                                 │
                        ┌────────▼────────┐
                        │   runs/compare/ │
                        │  summary.json   │
                        │  streaming_     │
                        │  curves.json    │
                        │  per_user_.csv  │
                        │  graphs/        │
                        └─────────────────┘
```

---

## Project File Map

```
csc-6711-final-project/
│
├── data/
│   ├── download_ml1m.py       Download MovieLens-1M from the web
│   ├── preprocess.py          Sort by timestamp, create warm/stream/test splits
│   └── drift_simulator.py     Swap user histories to simulate preference shift
│
├── static_mf.py               Model definition: Static Matrix Factorization
├── sequential_model.py        Model definition: GRU Sequential Recommender
├── bayesian_mf.py             Model definition: Bayesian Online MF
│
├── train_static_mf.py         Training script for Static MF
├── train_sequential.py        Training script for Sequential model
├── train_bayesian_mf.py       Training script for Bayesian MF
│
├── data_loader.py             Loads processed CSVs into Python objects
├── evaluate.py                Computes HR@K, NDCG@K, RMSE, MAE, Brier, ECE
├── streaming_eval.py          Replays interactions live, evaluates at checkpoints
├── compare_recommenders.py    Orchestrates the full head-to-head comparison
├── plot_results.py            Generates bar charts and streaming NDCG curves
│
├── *.sbatch                   SLURM job scripts for Rosie (MSOE supercomputer)
├── submit_recsys_pipeline.sh  Submit all 4 jobs (3 train + 1 compare) in order
│
└── runs_smoke/                Small-scale smoke test results (6 users)
    ├── static_mf/             Saved model + training history
    ├── sequential/            Saved model + training history
    ├── bayesian_mf/           Saved model + training history
    └── compare/               Head-to-head results + graphs
```

---

## Running the Project

### Locally (small smoke test)

```bash
# 1. Get the data
python data/download_ml1m.py

# 2. Build chronological splits
python data/preprocess.py \
  --input-dir data/raw/ml-1m \
  --output-dir data/processed

# 3. Train all three models
python train_static_mf.py     --data-dir data/processed --output-dir runs/static_mf
python train_sequential.py    --data-dir data/processed --output-dir runs/sequential
python train_bayesian_mf.py   --data-dir data/processed --output-dir runs/bayesian_mf

# 4. Run the comparison
python compare_recommenders.py \
  --data-dir data/processed \
  --static-checkpoint      runs/static_mf/static_mf.pt \
  --sequential-checkpoint  runs/sequential/sequential.pt \
  --bayesian-checkpoint    runs/bayesian_mf/bayesian_mf.pt \
  --output-dir runs/compare
```

### On Rosie (SLURM, full dataset)

```bash
# Submits all jobs with automatic dependencies
# (compare only starts once all 3 training jobs finish)
bash submit_recsys_pipeline.sh

# Common overrides:
PARTITION=teaching \
ENV=/data/csc4611/conda-csc4611 \
DATA_DIR=data/processed \
bash submit_recsys_pipeline.sh
```

---

## What We Expect to Find

```
Performance on ALL users:
  Static MF  ████████████░░░  decent baseline
  Sequential ██████████████░  better (uses history order)
  Bayesian   ███████████████  best or competitive

Performance on DRIFT users (preference changed):
  Static MF  ████████░░░░░░░  worse — stuck with old fingerprint
  Sequential ██████████░░░░░  better — reads recent history
  Bayesian   █████████████░░  best — actively updates its belief
```

The Bayesian model's advantage is expected to be most visible in the **drift user subset**, since that's the exact scenario it was designed for. On stable users, all three models may perform similarly.

---

## Key Technical Concept: What "Bayesian" Means Here

Instead of saying "this user's preference vector is exactly [0.2, 0.8, 0.1, ...]", the Bayesian model says "this user's preference vector is probably near [0.2, 0.8, 0.1, ...] but could also be [0.3, 0.7, 0.2, ...]" — it keeps track of its own uncertainty.

When you rate a new movie:
1. The model computes how surprising that rating was given its current belief
2. It updates the belief to make that rating less surprising in the future
3. The "forgetting factor" slightly weakens old evidence so recent ratings matter more

This is mathematically equivalent to running a Kalman filter on user preferences — the same algorithm used in GPS tracking and robotics to maintain a belief about a moving target.

---

## Output Files Explained

| File | What's in it |
|---|---|
| `summary.json` | Final HR@K, NDCG@K, RMSE, MAE for each model |
| `drift_subset_summary.json` | Same metrics, but only for users whose preferences drifted |
| `streaming_curves.json` | How each model's performance changed over the stream |
| `streaming_curve.csv` | Same, in spreadsheet-friendly format |
| `per_user_results.csv` | Every test prediction: model, user, true rating, predicted rating, error |
| `graphs/head_to_head_bar.png` | Bar chart comparing all three models |
| `graphs/streaming_ndcg10.png` | NDCG@10 over time as the stream progresses |
