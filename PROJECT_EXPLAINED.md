# CSC-6711 Final Project — Complete Visual Guide

> **Core question:** Do "belief-aware" recommenders that continuously update user profiles outperform systems that freeze profiles after training?

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Repository Structure](#2-repository-structure)
3. [The Problem](#3-the-problem)
4. [The Three Models](#4-the-three-models)
5. [Data Pipeline](#5-data-pipeline)
6. [Training](#6-training)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Full Execution Pipeline](#8-full-execution-pipeline)
9. [SLURM on Rosie](#9-slurm-on-rosie)
10. [Results](#10-results)
11. [How It All Connects](#11-how-it-all-connects)

---

## 1. The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   HYPOTHESIS                                                            │
│                                                                         │
│   Netflix/Spotify snapshot your taste once and never change their       │
│   mental model of you. But tastes drift — you went through a horror     │
│   phase, now you're into documentaries. A system that treats your       │
│   preferences as a living probability distribution and updates it       │
│   after every new rating should outperform frozen models, especially    │
│   when preferences shift.                                               │
│                                                                         │
│   We test this with three competing models on MovieLens-1M.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**The Three Contenders**

```
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│   Static MF          │    │   Sequential (GRU)    │    │   Bayesian Online MF │
│                      │    │                       │    │                      │
│   Classic matrix     │    │   Neural RNN that     │    │ ★ Proposed approach  │
│   factorization.     │    │   reads watch         │    │                      │
│   Learns fixed user/ │    │   history in order.   │    │   Treats preferences │
│   movie fingerprints.│    │   Context-aware but   │    │   as probability      │
│   Never updates.     │    │   weights never       │    │   distributions.     │
│                      │    │   change.             │    │   Updates via Bayes  │
│   BASELINE           │    │   COMPARISON          │    │   after every rating.│
│   Expected worst     │    │   Should beat Static  │    │   Expected to excel  │
│   on drift users     │    │   but lag on drift    │    │   on drift users     │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

---

## 2. Repository Structure

```
csc-6711-final-project/
│
├── data/                          ← Data acquisition & processing
│   ├── download_ml1m.py           ← Downloads MovieLens-1M from GroupLens
│   ├── preprocess.py              ← Filters, splits, maps IDs to 0-based indices
│   ├── drift_simulator.py         ← Injects synthetic preference drift
│   ├── raw/ml-1m/                 ← ratings.dat, movies.dat (raw)
│   └── processed/                 ← interactions.csv, metadata.json, id maps
│
├── static_mf.py                   ← Model definition: StaticMF (nn.Module)
├── sequential_model.py            ← Model definition: SequentialRecommender (GRU)
├── bayesian_mf.py                 ← Model definition: BayesianMFState (no grad)
│
├── train_static_mf.py             ← Trains Static MF (MSE or BPR loss)
├── train_sequential.py            ← Trains Sequential (item + aux heads)
├── train_bayesian_mf.py           ← Trains warm Static MF → exports as Bayesian init
│
├── data_loader.py                 ← Datasets, collators, load_processed_data()
├── evaluate.py                    ← HR@K, NDCG@K, RMSE, MAE, Brier, ECE
├── streaming_eval.py              ← Replays stream; calls runtime.update() per row
├── compare_recommenders.py        ← Main orchestrator: loads all 3 + runs eval
├── eval_popularity.py             ← Popularity baseline runtime
│
├── plot_results.py                ← Simple bar charts + streaming curves
├── generate_plots.py              ← Advanced multi-panel comparison plots
│
├── submit_recsys_pipeline.sh      ← SLURM orchestrator (submits all 4 jobs)
├── recsys_static_mf.sbatch        ← SLURM: train Static MF
├── recsys_sequential.sbatch       ← SLURM: train Sequential
├── recsys_bayesian_mf.sbatch      ← SLURM: train Bayesian MF
├── recsys_compare.sbatch          ← SLURM: compare (waits for all 3 to finish)
├── recsys_bayesian_tune.sbatch    ← SLURM: hyperparameter grid search
├── recsys_static_mf_bpr.sbatch    ← SLURM: BPR-loss variant
│
└── runs/                          ← All outputs
    ├── static_mf/                 ← static_mf.pt, history.json, summary.json
    ├── sequential/                ← sequential.pt, history.json, summary.json
    ├── bayesian_mf/               ← bayesian_mf.pt, history.json, summary.json
    ├── compare/                   ← summary.json, drift_subset_summary.json,
    │                              │  streaming_curves.json, per_user_results.csv
    ├── bayesian_tune/             ← Grid: f090_nv050/, f095_nv025/, ...
    ├── static_mf_bpr/             ← BPR-trained checkpoint
    └── plots/                     ← PNG figures
```

---

## 3. The Problem

### Dataset: MovieLens-1M

```
1,000,209 ratings   6,040 users   3,706 movies   Timestamps included
         │
         ▼
Per-user chronological split (60% / 30% / 10%)
         │
   ┌─────┼─────────────────────────────────┐
   │     │                                 │
   ▼     ▼                                 ▼
  WARM  STREAM                            TEST
  60%    30%                              10%
         │                                 │
  All 3 models         Bayesian/GRU        Final holdout
  trained here.        update live.        Never seen
                       All models          during training.
                       evaluated here
                       periodically.
```

### The Drift Simulation

To stress-test the Bayesian model in exactly the scenario it was designed for, preference drift is synthetically injected:

```
BEFORE DRIFT INJECTION:

  User A timeline:  ──[Action]──[Action]──[Thriller]──[Thriller]──▶
  User B timeline:  ──[Romance]──[Romance]──[Comedy]──[Comedy]────▶


AFTER DRIFT INJECTION (swap second half of stream):

  User A timeline:  ──[Action]──[Action]──|──[Comedy]──[Romance]──▶
  User B timeline:  ──[Romance]──[Romance]─|──[Thriller]──[Action]──▶
                                          ↑
                                     DRIFT POINT


EXPECTED OUTCOME:

  Static MF: frozen fingerprint → recommends Action for User A forever → ✗
  Bayesian:  belief updates after each rating → adapts to Comedy → ✓
```

`drift_simulator.py` selects 1,000 users, pairs them randomly, swaps the second half of their stream histories, and writes `drift_meta.json` to track which users experienced drift.

---

## 4. The Three Models

### 4a. Static Matrix Factorization (`static_mf.py`)

**Core idea:** Decompose the user-item rating matrix into low-rank factors.

```
        Items (3,706)
         ─────────────────────────────────
        │                               │
Users   │    ? 4 ? 1 ? ? 5 ? 3 ? ?     │
(6,040) │    ? ? 2 ? ? 4 ? ? ? ? 1     │
        │    3 ? ? ? 5 ? ? 2 ? ? ?     │
         ─────────────────────────────────
                        ↕
              MATRIX FACTORIZATION
                        ↕

User Matrix U (6040 × 64)    Item Matrix V (3706 × 64)
  Each row = user's          Each row = movie's
  latent taste fingerprint   latent genre fingerprint

Predicted rating = global_bias + user_bias[u] + item_bias[i] + dot(U[u], V[i])
                                                                  └─────────────┘
                                                                  Compatibility score
```

**Key classes:**

```
StaticMFConfig
├── num_users: int
├── num_items: int
├── embedding_dim: int = 64
└── global_mean: float

StaticMF(nn.Module)
├── user_factors:  Embedding(num_users, embedding_dim)
├── item_factors:  Embedding(num_items, embedding_dim)
├── user_bias:     Embedding(num_users, 1)
├── item_bias:     Embedding(num_items, 1)
├── forward(user_ids, item_ids) → predicted_ratings
├── predict_all_items(user_id)  → scores for every item
└── export_state()              → dict (CPU tensors)
```

**Two training objectives:**

```
MSE Loss (rating prediction)          BPR Loss (ranking)
─────────────────────────────         ────────────────────────────────────
L = (predicted - actual)²             For each (user, pos_item, neg_item):
                                      L = -log(σ(score_pos - score_neg))
Good for: RMSE                        Good for: HR@K, NDCG@K
Bad for:  HR@K, NDCG@K               Bad for:  RMSE
```

**Training defaults:** 8 epochs, batch_size=4096, lr=1e-3, weight_decay=1e-6

---

### 4b. Sequential Recommender (`sequential_model.py`)

**Core idea:** Read the user's watch history in temporal order with a GRU; predict what they'll watch next.

```
User's watch history (ordered by time):
  [Titanic] [Saving Private Ryan] [The Matrix] [? → predict this]
      ↓              ↓                  ↓
  item_emb(64)   item_emb(64)      item_emb(64)   ← movie embedding lookup
  + rating_emb   + rating_emb      + rating_emb   ← embed the given rating (1-5)
      │              │                  │
      └──────────────┴──────────────────┘
                       │
                      GRU
                  (hidden_dim=128)
                       │
                   hidden state
                       │
            ┌──────────┼───────────────┐
            │          │               │
        next_item    genre_head    rating_head
        projection   (BCE loss)   (CE loss)
            │        (20 genres)  (5 buckets)
            ↓
        dot product with all item embeddings
            ↓
        ranked item scores → top-10 recommendation
```

**Auxiliary heads** (genre + rating prediction) force the GRU to encode richer signals:

```
WITHOUT aux heads:  GRU only sees item sequences → learns "what comes after what"
WITH aux heads:     GRU must also predict genre & rating → learns "why" users watch things
```

**Key methods:**

```
SequentialRecommender(nn.Module)
├── encode_history(item_hist, rating_hist, lengths) → hidden_state
├── forward(...)                                    → (item_logits, genre_logits, rating_logits)
├── score_candidates(user_history, candidate_items) → scores
└── [training only] auxiliary_loss() combines all three losses
```

**Training defaults:** 5 epochs, batch_size=512, lr=1e-3, dropout=0.1, grad_clip=1.0

---

### 4c. Bayesian Online Matrix Factorization (`bayesian_mf.py`)

**Core idea:** Treat each user's preferences as a probability distribution. Update the distribution with Bayes' rule after every new rating. Old evidence fades via a forgetting factor.

```
TRADITIONAL MF:
  User fingerprint = single point in 64-dimensional space
  ●  ← fixed forever after training


BAYESIAN MF:
  User fingerprint = Gaussian distribution in 64-dimensional space
  🔵  ← wide uncertainty = know little about user
  🔵  ← narrows as ratings arrive, shifts when preferences change

  Before any ratings:    After 10 ratings:    After drift:
      🔵 (wide)              🔵 (narrow)          🔵 (shifted)
```

**The Bayes Update (after each new rating):**

```
New rating arrives: user u gives item i a rating r
                         │
                    q = item_factors[i]       ← item's fingerprint
              residual = r - global_bias - item_bias[i] - q·μ_prev
                         │
   PRECISION UPDATE:     │
   Λ_new = λ·Λ_prev + (1/σ²)·q⊗q            ← sharpen belief at q direction
                         │
   ETA UPDATE:           │
   η_new = λ·η_prev + (1/σ²)·residual·q      ← shift belief toward new evidence
                         │
   POSTERIOR MEAN:   μ_new = Λ_new⁻¹·η_new   ← solve linear system

   Where:
     λ = forgetting factor (0.9–1.0); <1.0 fades old evidence
     σ² = noise_var; controls update magnitude
```

This is mathematically equivalent to a **Kalman filter** — the same algorithm used in GPS and robotics.

**Key structure:**

```
BayesianMFState (dataclass, no nn.Module — pure tensor math)
├── item_factors:    Tensor(num_items, dim)    ← fixed from warm training
├── item_bias:       Tensor(num_items)         ← fixed from warm training
├── global_bias:     float
├── user_precision:  Tensor(num_users, dim, dim)  ← Λ, inverse covariance
├── user_eta:        Tensor(num_users, dim)       ← Λμ, precision-weighted mean
├── noise_var:       float = 0.5
├── forgetting:      float = 1.0
│
├── user_mean(user_id) → Tensor(dim)
│     Solves: Λ @ μ = η  (torch.linalg.solve)
│
├── predict_rating(user_id, item_id) → (mean: float, variance: float)
│     mean     = global_bias + item_bias[i] + q·μ
│     variance = q @ Λ⁻¹ @ q + σ²
│
├── score_items(user_id, item_ids) → Tensor
│     Vectorized mean predictions over candidate set
│
└── update(user_id, item_id, rating) → modifies precision & eta in-place
```

**Initialization:** Bayesian warm training = train a standard Static MF on warm data, then:
- Fix item factors and biases
- Set user prior mean = learned user embeddings
- Set precision = (1/prior_var) × Identity (broad initial belief)
- No separate Bayesian training step needed

---

## 5. Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Download                                                        │
│                                                                          │
│  download_ml1m.py                                                        │
│    ↓                                                                     │
│  data/raw/ml-1m/                                                         │
│    ├── ratings.dat   (UserID::MovieID::Rating::Timestamp)                │
│    └── movies.dat    (MovieID::Title::Genres)                            │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Preprocess (preprocess.py)                                      │
│                                                                          │
│  Filters: keep users with ≥20 interactions                               │
│  Maps: raw MovieLens IDs → contiguous 0-based integers                   │
│  Splits: per-user chronological 60% warm / 30% stream / 10% test        │
│  Genre encoding: 18 binary genre flags per movie                         │
│                                                                          │
│  Output → data/processed/                                                │
│    ├── interactions.csv      (user_id, item_id, rating, timestamp, split)│
│    ├── metadata.json         (counts, global_mean_rating)                │
│    ├── user_id_map.json      (raw→new)                                   │
│    ├── item_id_map.json      (raw→new)                                   │
│    └── item_genres.json      (genre_names, item→genres)                  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3 (optional): Drift Simulation (drift_simulator.py)                │
│                                                                          │
│  Select 1,000 stream users → pair them → swap second-half histories     │
│                                                                          │
│  Output:                                                                 │
│    ├── interactions_drift.csv    (modified interactions)                 │
│    └── drift_meta.json           (which users, which pairs)              │
└─────────────────────────────────────────────────────────────────────────┘
```

### data_loader.py: What Gets Loaded

```
load_processed_data(data_dir) returns a dict with:

  ┌──────────────────────────────────────────────────────────┐
  │  metadata          num_users, num_items, global_mean     │
  │  interactions      All rows sorted by (timestamp)        │
  │  by_split          { 'warm': [...], 'stream': [...],     │
  │                      'test': [...] }                     │
  │  by_user           List of interactions per user         │
  │  seen_items_by_user  Set of watched items per user       │
  │  item_to_genres    Dict[item_id → List[genre_id]]        │
  │  genre_names       ['Action', 'Animation', ...]          │
  └──────────────────────────────────────────────────────────┘

PyTorch Datasets:

  MFTrainDataset           → (user_id, item_id, rating) triplets
  BPRTrainDataset          → (user_id, pos_item, neg_item) triplets
  NextItemSequenceDataset  → (prefix_history, next_item_target) windows
```

---

## 6. Training

### Static MF Training (`train_static_mf.py`)

```
warm_interactions
      │
      ├─ MFTrainDataset / BPRTrainDataset
      │
      ▼
   [batch: (user, item, rating)]
      │
   StaticMF.forward()
      │
   MSE loss: (pred - rating)²       OR    BPR loss: -log(σ(score_pos - score_neg))
      │
   Adam optimizer step
      │
   (repeat 8 epochs)
      │
      ▼
  static_mf.pt
  history.json  (train loss per epoch)
  summary.json  (config + final loss)
```

### Sequential Training (`train_sequential.py`)

```
warm_interactions
      │
      ├─ NextItemSequenceDataset
      │    Each sample: (item_prefix, rating_prefix, next_item_target,
      │                  next_genre_target, next_rating_bucket_target)
      │
      ▼
   SequentialRecommender.forward()
      │
   total_loss = item_loss (CE)
              + 0.1 × genre_loss (BCE)
              + 0.1 × rating_loss (CE)
      │
   Adam + gradient clipping (max_norm=1.0)
      │
   (repeat 5 epochs)
      │
      ▼
  sequential.pt
```

### Bayesian Warm Training (`train_bayesian_mf.py`)

```
warm_interactions
      │
      ├─ (same pipeline as Static MF MSE)
      │
      ▼
  Train a regular Static MF (8 epochs)
      │
      ▼
  Extract:  item_factors, item_bias, global_bias, user_embeddings
      │
      ▼
  Build BayesianMFState:
    ├── item_factors  ← copied from Static MF (FIXED)
    ├── item_bias     ← copied from Static MF (FIXED)
    ├── user_precision[u] = (1/prior_var) × Identity_64   (for all u)
    └── user_eta[u]   = precision × user_embedding[u]
      │
      ▼
  bayesian_mf.pt
  (no additional Bayesian training; warm-up is the Static MF phase)
```

---

## 7. Evaluation Framework

### The Ranking Protocol

For every test interaction `(user, true_item, rating)`:

```
1. Sample 100 random items user has NOT seen → "negatives"
2. Pool: 1 true_item + 100 negatives = 101 candidates
3. Score all 101 with runtime.score_items(user, candidates)
4. Rank by descending score
5. Find position of true_item in the ranked list

   If true_item is at rank r:
   ├── HR@K   = 1 if r ≤ K else 0
   └── NDCG@K = 1/log₂(r+1) if r ≤ K else 0

K values tested: 5, 10, 20
Primary metric: HR@10 and NDCG@10
```

### Streaming Evaluation (`streaming_eval.py`)

```
Stream rows (chronological order):
  t=1: user_42 watches Movie_123 (rating: 4)
  t=2: user_07 watches Movie_456 (rating: 3)
  t=3: ...
           │
           ├─ BayesianRuntime.update(row)   → Bayes update on user_42's belief
           ├─ SequentialRuntime.update(row) → append to user_42's history buffer
           └─ StaticRuntime.update(row)     → no-op (frozen)
           │
  Every 20,000 stream steps:
           │
           ▼
    evaluate all runtimes on test set
    record StreamingPoint:
      { stream_step, hr@10, ndcg@10, rmse, mae, brier, ece }
           │
           ▼
  streaming_curves.json
```

### Metrics Reference

```
┌─────────────────┬───────────────────────────────────────────────────────────┐
│ Metric          │ Meaning                                                   │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ HR@10           │ % of test items that appear in top-10 recommendations     │
│                 │ Binary: hit (1) or miss (0)                               │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ NDCG@10         │ Ranking quality — higher reward for items ranked near top │
│                 │ Score = 1/log₂(rank+1), 0 if rank > 10                  │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ RMSE / MAE      │ Rating prediction error (lower = better)                 │
│                 │ RMSE penalizes large errors more                          │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Brier Score     │ Bayesian-only. Mean squared error of predicted            │
│                 │ probability distributions over rating buckets             │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ ECE             │ Bayesian-only. Expected Calibration Error — are the       │
│                 │ model's confidence intervals actually reliable?           │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

### Runtime Wrappers (`compare_recommenders.py`)

All three models expose the same interface via a `RecommenderRuntime` protocol:

```
class RecommenderRuntime(Protocol):
    def score_items(self, user_id, item_ids) → Tensor   # higher = more recommended
    def predict_rating(self, user_id, item_id) → float  # predicted star rating
    def update(self, interaction) → None                 # called during streaming

StaticRuntime    → score_items: dot(U[u], V[i]) + biases
                   update: no-op

SequentialRuntime → score_items: GRU(history) → dot with item embs
                    update: append (item, rating) to user's history buffer

BayesianRuntime  → score_items: q·μ_u + biases (posterior mean)
                   update: Bayes rule on user_u's precision & eta
```

---

## 8. Full Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOCAL SMOKE TEST                                │
└─────────────────────────────────────────────────────────────────────────┘

python data/download_ml1m.py
  └─→ data/raw/ml-1m/

python data/preprocess.py --input-dir data/raw/ml-1m --output-dir data/processed
  └─→ data/processed/

python data/drift_simulator.py ...
  └─→ data/processed/interactions_drift.csv + drift_meta.json

python train_static_mf.py   --data-dir data/processed --output-dir runs/static_mf
  └─→ runs/static_mf/static_mf.pt

python train_sequential.py  --data-dir data/processed --output-dir runs/sequential
  └─→ runs/sequential/sequential.pt

python train_bayesian_mf.py --data-dir data/processed --output-dir runs/bayesian_mf
  └─→ runs/bayesian_mf/bayesian_mf.pt

python compare_recommenders.py \
  --data-dir data/processed \
  --static-checkpoint     runs/static_mf/static_mf.pt \
  --sequential-checkpoint runs/sequential/sequential.pt \
  --bayesian-checkpoint   runs/bayesian_mf/bayesian_mf.pt \
  --output-dir            runs/compare
  └─→ runs/compare/{summary.json, streaming_curves.json,
                    drift_subset_summary.json, per_user_results.csv}

python generate_plots.py --summary-json runs/compare/summary.json ...
  └─→ runs/plots/*.png


┌─────────────────────────────────────────────────────────────────────────┐
│                        ROSIE (FULL SCALE)                               │
└─────────────────────────────────────────────────────────────────────────┘

PARTITION=teaching ENV=/data/csc4611/conda-csc4611 DATA_DIR=data/processed \
bash submit_recsys_pipeline.sh

  submit_recsys_pipeline.sh
    │
    ├─ sbatch recsys_static_mf.sbatch      → job_id_1  (1 V100, ~1hr)
    ├─ sbatch recsys_sequential.sbatch     → job_id_2  (1 V100, ~1hr)
    ├─ sbatch recsys_bayesian_mf.sbatch    → job_id_3  (1 V100, ~1hr)
    │
    └─ sbatch --dependency=afterok:job_id_1:job_id_2:job_id_3 \
              recsys_compare.sbatch        → runs compare_recommenders.py
```

---

## 9. SLURM on Rosie

### Job Dependency Graph

```
         ┌─────────────────────────────────────────────────┐
         │           submit_recsys_pipeline.sh             │
         └──────────┬───────────────┬────────────┬─────────┘
                    │               │            │
                    ▼               ▼            ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │static_mf.    │ │sequential.   │ │bayesian_mf.  │
         │sbatch        │ │sbatch        │ │sbatch        │
         │              │ │              │ │              │
         │ 1 V100 GPU   │ │ 1 V100 GPU   │ │ 1 V100 GPU   │
         │ 8 CPUs       │ │ 8 CPUs       │ │ 8 CPUs       │
         │ 64GB RAM     │ │ 64GB RAM     │ │ 64GB RAM     │
         │ 1-day limit  │ │ 1-day limit  │ │ 1-day limit  │
         └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                │                │                 │
                └────────────────┴─────────────────┘
                                 │
                    afterok (all 3 must succeed)
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  compare.sbatch  │
                       │                  │
                       │ No GPU needed    │
                       │ 8 CPUs, 64GB RAM │
                       │ 1-day limit      │
                       └──────────────────┘
```

### Additional Job Scripts

```
Hyperparameter tuning:
  submit_bayesian_tuning.sh → recsys_bayesian_tune.sbatch
    Tests: forgetting ∈ {0.90, 0.95, 0.98}, noise_var ∈ {0.25, 0.50}
    Grid creates: runs/bayesian_tune/f090_nv050/, f095_nv025/, f095_nv050/, f098_nv050/

BPR variant:
  submit_static_mf_bpr.sh → recsys_static_mf_bpr.sbatch
    Trains Static MF with BPR ranking loss instead of MSE
    Output: runs/static_mf_bpr/static_mf.pt
```

---

## 10. Results

### Full Test Set (102,759 ratings)

```
                HR@10       NDCG@10     RMSE
               ──────────  ──────────  ──────
Sequential  ▓▓▓▓▓▓▓▓▓▓  0.535  ▓▓▓▓▓▓  0.328  (no RMSE — ranking model)
Static BPR  ▓▓▓▓▓▓▓▓    0.453  ▓▓▓▓    0.239  (no RMSE — ranking model)
Popularity  ▓▓▓▓▓▓▓     0.411  ▓▓▓▓    0.218    1.158
Static MSE  ▓▓▓▓        0.250  ▓▓      0.133    0.926  ← best RMSE
Bayesian    ▓▓▓         0.222  ▓▓      0.117    1.023
```

**Winner:** Sequential dominates on ranking metrics.

### Drift Users Only (17,811 ratings, 1,000 swapped users)

```
                HR@10 (drift)   NDCG@10 (drift)
               ──────────────   ───────────────
Sequential  ▓▓▓▓▓▓▓▓▓▓  0.535  ▓▓▓▓▓▓  0.325
Static BPR  ▓▓▓▓▓▓▓▓    0.442  ▓▓▓▓    0.235
Popularity  ▓▓▓▓▓▓▓     0.404  ▓▓▓▓    0.216
Static MSE  ▓▓▓▓        0.247  ▓▓      0.132
Bayesian    ▓▓▓         0.217  ▓▓      0.113
```

**Hypothesis result:** Bayesian's expected advantage on drift users did NOT materialize.

### Streaming Curves

```
HR@10 over stream time:

0.55 │                                             ╭───── Sequential
     │                                        ╭───╯
0.45 │                                   ╭───╯
     │                              ╭───╯
0.35 │                         ╭───╯
     │                    ╭───╯
     │               ╭───╯
0.25 │──────────────────────────────────────────── Static MF (flat)
     │──────────────────────────────────────────── Bayesian  (flat)
0.20 │
     └────────────────────────────────────────────▶
     0K         100K        200K        300K   stream steps
```

Sequential's HR@10 climbs from 0.32 → 0.53 as it accumulates history context.  
Static MF and Bayesian are flat — their scoring doesn't improve with more streaming data.

### Why Bayesian Underperformed

```
ROOT CAUSE ANALYSIS:

Training objective:   MSE (minimize rating prediction error)
Evaluation metric:    HR@10 / NDCG@10 (ranking quality)

These optimize fundamentally different things:

  MSE optimizes: "predict the exact star rating accurately"
  HR@10 needs:   "rank the correct movie #1 out of 100 candidates"

The Bayesian model inherited Static MF's MSE training objective.
Even with perfect Bayes updates, if the base scores are miscalibrated
for ranking, the updates won't fix the ranking problem.

CONFIRMED BY: Static MF (BPR) >> Static MF (MSE) on HR@10
              Using ranking loss alone closes most of the gap.

RECOMMENDED FIX: Train Bayesian model's warm phase with BPR loss.
```

---

## 11. How It All Connects

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE DATA FLOW                                    │
└────────────────────────────────────────────────────────────────────────────────┘

MovieLens-1M (raw)
      │  download_ml1m.py
      ▼
data/raw/ml-1m/
      │  preprocess.py
      ▼
data/processed/
  interactions.csv ──────────────────────────────────────────────────┐
  metadata.json                                                       │
  item_genres.json                                                    │
      │  drift_simulator.py                                           │
      ▼                                                               │
  interactions_drift.csv + drift_meta.json                           │
      │                                                               │
      │     data_loader.py (load_processed_data)                      │
      ▼                                                               │
  warm_rows ──┬──────────────────────────────────────┐               │
              │                                       │               │
              ▼                                       ▼               │
     train_static_mf.py               train_bayesian_mf.py           │
     train_sequential.py                     │                        │
              │                 (trains Static MF internally,         │
              │                  extracts item factors)               │
              ▼                              ▼                        │
     static_mf.pt              sequential.pt   bayesian_mf.pt        │
              │                      │               │                │
              └──────────────────────┴───────────────┘                │
                                     │                                 │
                              compare_recommenders.py ←───────────────┘
                              (loads all 3 checkpoints
                               + data including drift_meta)
                                     │
                    ┌────────────────┼───────────────────┐
                    │                │                   │
                    ▼                ▼                   ▼
             StaticRuntime  SequentialRuntime  BayesianRuntime
             (frozen)       (history buffer)  (belief updater)
                    │                │                   │
                    └────────────────┴───────────────────┘
                                     │
                              streaming_eval.py
                              (replay stream,
                               call runtime.update(),
                               checkpoint at 20K steps)
                                     │
                              evaluate.py
                              (HR@K, NDCG@K, RMSE,
                               Brier, ECE per model)
                                     │
                    ┌────────────────┼───────────────────┐
                    ▼                ▼                   ▼
             summary.json  streaming_curves.json  drift_subset_summary.json
             per_user_results.csv
                    │
             generate_plots.py / plot_results.py
                    │
             runs/plots/*.png
```

---

### Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Sequential (GRU) is the strongest model — context from history       │
│     helps more than belief updates on this dataset.                     │
│                                                                         │
│  2. Loss function matters more than model architecture for ranking:     │
│     BPR-trained Static MF beats MSE-trained Bayesian, despite the      │
│     Bayesian model being more sophisticated.                            │
│                                                                         │
│  3. Bayesian updates are mathematically sound (Kalman filter) but       │
│     need to be combined with a ranking-aware training objective         │
│     to actually win on HR@10 / NDCG@10.                                │
│                                                                         │
│  4. The drift injection didn't reveal a Bayesian advantage because      │
│     the ranking calibration problem dominated the drift signal.         │
│                                                                         │
│  5. Next logical step: train Bayesian warm phase with BPR loss,         │
│     then apply Bayesian online updates — this should be the best        │
│     of both worlds.                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```
