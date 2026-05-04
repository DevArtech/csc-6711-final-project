# SkyNet: Belief-Aware Planning in Partially Observable Stochastic Games

> **arXiv:** 2603.27751  
> **One-sentence summary:** Adding outcome-prediction auxiliary heads to MuZero — without any explicit belief tracking — is enough to make it decisively stronger in a hidden-information card game, but only once training data flow is sufficient.

---

## Why This Paper Matters to This Project

The recommender system project draws directly on the central idea here: **maintaining a living, uncertain belief about a hidden quantity** (user preferences / hidden card values) and updating it incrementally rather than fixing it after an initial training pass. The paper provides rigorous empirical evidence that this approach works and shows *why* it works through latent representation analysis.

---

## 1. The Problem

Classic MuZero is outstanding at games where both players can see the full board state at all times (Chess, Go, Atari). Real-world decisions rarely work that way. Consider:

```
Perfect-information game (Chess):
  Agent observes: EVERYTHING
  Hidden state:   nothing
  Strategy:       plan ahead with certainty

Partially observable game (Skyjo card game):
  Agent observes: its own cards, top of discard pile, opponents' face-up cards
  Hidden state:   face-down cards, full deck composition, opponents' unknown cards
  Strategy:       must reason under uncertainty about what it cannot see
```

Standard MuZero has no mechanism to represent "I don't know what's there." It treats its latent embedding as a point estimate of state, not a distribution. This is fine when you can see everything; it breaks down when you can't.

---

## 2. The Skyjo Testbed

The paper builds and uses **Skyjo** — a partially observable, stochastic, non-zero-sum, 2–8 player card game — as the evaluation domain.

```
Deck: 150 cards valued −2 to +12
      ┌────────────────────────────────┐
      │  -2 × 5    0 × 15             │
      │  -1 × 10   +1 through +12 × 10│
      └────────────────────────────────┘

Each player's grid (3 rows × 4 columns = 12 cards):

  ┌───┬───┬───┬───┐
  │ ? │ 7 │ ? │ 3 │   ? = face-down (hidden from everyone)
  ├───┼───┼───┼───┤   number = revealed card
  │ 4 │ ? │ 9 │ ? │
  ├───┼───┼───┼───┤
  │ ? │ 2 │ ? │ 6 │
  └───┴───┴───┴───┘

Goal: lowest cumulative score when any player hits 100 points total.
Bonus: three matching cards in a column → column removed (0 score).
Trap:  if you reveal your last card but don't have the lowest score → score doubles.
```

### Why Skyjo is a Good Testbed

| Property | Why It Matters |
|---|---|
| Hidden cards | Agent must reason about unobserved information |
| Stochastic deck draws | Outcomes are probabilistic, not deterministic |
| Non-zero-sum scoring | One player's loss ≠ another's gain (unlike Chess) |
| 2–8 players | Multi-player dynamics and opponent modeling |
| Sparse win signal | ~1/N win rate makes learning harder |

---

## 3. The Architecture: Baseline vs. SkyNet

Both models share the same core MuZero structure. SkyNet adds two components.

### 3.1 Core MuZero (both models)

```
                    ┌──────────────────────────────────────────┐
                    │           REPRESENTATION NETWORK          │
                    │                                          │
  Observation ────► │  Token embeddings → Transformer (6L/8H) │ ──► h₀ ∈ ℝ⁵¹²
  (partial obs)     │  Board + discard + global + history      │
                    └──────────────────────────────────────────┘
                                         │
                                         │ latent state h₀
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │            DYNAMICS NETWORK               │
                    │                                          │
  Action a_k ─────► │  MLP: concat(h_k, embed(a_k)) → h_{k+1} │ ──► h_{k+1}, r_k
                    │  Residual connection + LayerNorm          │     (next state,
                    └──────────────────────────────────────────┘      reward)
                                         │
                                         │ latent state h_k
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │           PREDICTION NETWORK              │
                    │                                          │
                    │  Policy head:  π_k  (action logits)      │
                    │  Value head:   v_k  (expected return)     │
                    └──────────────────────────────────────────┘
```

MCTS runs entirely in this latent space — no game simulator needed at inference time.

### 3.2 What SkyNet Adds

```
                                      h_k (latent state)
                                           │
                    ┌──────────────────────┼───────────────────────┐
                    │         EGO CONDITIONING LAYER                │
                    │                                               │
                    │  h_cond = LayerNorm( h_k                      │
                    │                    + e_ego        ← who am I? │
                    │                    + e_current    ← whose turn │
                    │                    + e_nplayers ) ← how many  │
                    └──────────────────────┬───────────────────────┘
                                           │ h_cond
                          ┌────────────────┼────────────────┐
                          │                │                │
                          ▼                ▼                ▼
                    ┌──────────┐   ┌──────────┐   ┌──────────────┐
                    │  Policy  │   │  Value   │   │  AUXILIARY   │
                    │  head    │   │  head    │   │  HEADS ★     │
                    │          │   │          │   │              │
                    │  π_k     │   │  v_k     │   │ Winner head: │
                    │ (actions)│   │(ego ret.)│   │ P(player i   │
                    └──────────┘   └──────────┘   │  wins)       │
                                                  │              │
                                                  │ Rank head:   │
                                                  │ E[final rank │
                                                  │  per player] │
                                                  └──────────────┘
```

The auxiliary heads don't change how MCTS plans. They change what the latent state *has to encode* to minimize the training loss — forcing it to retain outcome-relevant hidden-state information.

### 3.3 Side-by-Side Architecture Comparison

```
┌──────────────────────────────────┬──────────────────────────────────┐
│       BASELINE MUZERO            │           SKYNET                 │
├──────────────────────────────────┼──────────────────────────────────┤
│ Representation: Transformer      │ Same                             │
│ Dynamics: MLP + residual         │ Same                             │
│ Prediction: policy + value       │ policy + value +                 │
│                                  │   winner head + rank head        │
│ Conditioning: none               │ Ego conditioning on all heads    │
│ Loss: π + v + r + L2             │ π + v + r + L2 +                 │
│                                  │   α·winner + β·rank              │
│ Perspective: single agent        │ Perspective augmentation:        │
│                                  │   each trajectory → N examples   │
│                                  │   (one per player viewpoint)     │
└──────────────────────────────────┴──────────────────────────────────┘
```

---

## 4. Training Pipeline

### 4.1 Decision-Granularity Action Decomposition

A standard MuZero agent would pick "draw a card and place it" as one atomic action. But the card value is unknown at decision time. SkyNet decomposes each turn into sequential micro-decisions:

```
Turn flow:

  ┌─────────────────────────────────────────────────────────────┐
  │  PHASE A: Choose source (2 actions)                         │
  │                                                             │
  │  [Draw from deck]      [Take from discard pile]             │
  │         │                        │                          │
  │         │ card revealed          │ card already known       │
  │         ▼                        ▼                          │
  │  PHASE B: Keep or discard? (2 actions)          ← deck only │
  │                                                             │
  │  [Keep drawn card]     [Discard drawn card]                 │
  │         │                        │                          │
  │         │                        │                          │
  │         ▼                        ▼                          │
  │  PHASE C: Choose grid position (12 actions)                 │
  │                                                             │
  │  [Replace slot 0..11]  [Flip face-down slot 0..11]          │
  └─────────────────────────────────────────────────────────────┘

  Total masked action space: 16 actions
  Legal actions depend on current phase
```

This matters because MCTS can now branch meaningfully at each decision point, with correct information available at each step.

### 4.2 Curriculum and Opponent Pool

```
Training opponents (heuristic bots):
  ┌────────────────────────────────────────────────────────────┐
  │  1. Greedy value replacement  (always replace highest card) │
  │  2. Information-first flip    (prefer flipping unknowns)    │
  │  3. Column hunter             (aims for matching columns)   │
  │  4. Risk-aware replacement    (weights expected card value) │
  │  5. End-round aggro           (rushes to end rounds)        │
  │  6. Anti-discard              (avoids feeding opponents)    │
  └────────────────────────────────────────────────────────────┘

Opponent sampling during self-play:
  70% current policy  ←── prevents collapse to one strategy
  30% random from checkpoint pool  ←── prevents oscillation
```

### 4.3 MCTS Simulation Schedule

```
Simulation budget increases over training:

Iterations:    0────────200────────500────────────────►
               │         │          │
Sims/move:   200        400        600

Rationale: more compute buys better policy targets;
           ramp avoids wasting budget when policy is random early on
```

### 4.4 Auxiliary Loss Ramp

```
Auxiliary loss weights ramp up gradually:

α (winner loss): 0.1 ──────────────────────► 0.5
β (rank loss):   0.1 ──────────────────────► 0.25
                 │                            │
              early training              late training
              (primary MuZero            (auxiliary tasks
               objective                  fully weighted)
               dominates)

Total loss:  ℒ = ℒ_MuZero + α·ℒ_winner + β·ℒ_rank

Prevents auxiliary tasks from dominating when data is scarce
and latent representations are still forming.
```

---

## 5. Results

### 5.1 Head-to-Head Performance Over Training

1000-game matches at each checkpoint, alternating seats:

```
Win rate of SkyNet vs. Baseline MuZero:

100% │
     │                                      ████
 75% │                               ████  █   █
     │                         ████ █    ██     █──── 75.3% (iter 1000)
 50% │──────────────────────────────────────────── (even)
     │               ████
 42% │         ████ █
 36% │  ████  █
     │ █    ██
  0% │
     └─────────────────────────────────────────────►
     iter: 125    250    500    750   1000

     ◄── SkyNet worse ──┤── SkyNet better ──►
                      crossover
                    ~iter 250–500
```

| Checkpoint | SkyNet WR | 95% CI | Δ Elo | Significance |
|---|---|---|---|---|
| Iter 125 | 36.0% | [33.1–39.0%] | −99 | SkyNet **worse** |
| Iter 250 | 42.2% | [39.2–45.3%] | −55 | SkyNet worse |
| **Iter 500** | **66.8%** | **[63.8–69.6%]** | **+120** | **p < 10⁻²⁵** |
| Iter 750 | 74.2% | [71.4–76.8%] | +184 | p < 10⁻⁵⁰ |
| **Iter 1000** | **75.3%** | **[72.5–77.9%]** | **+194** | **p < 10⁻⁵⁰** |

### 5.2 Performance Against Heuristic Bots

```
Win rate vs. curriculum heuristic opponents:

Belief-Aware (SkyNet): ████████████████████████████████░░░  0.720
Baseline MuZero:       ████████████████████░░░░░░░░░░░░░░░  0.466
                       0                 0.5                 1.0

SkyNet minimum (0.525) > Baseline mean (0.466)
→ Worst SkyNet checkpoint beats average Baseline checkpoint
```

| Metric | Baseline | SkyNet |
|---|---|---|
| Mean eval win rate | 0.466 | **0.720** |
| Max win rate | 0.600 | **0.825** |
| Min win rate | 0.313 | **0.525** |
| Mean episode length | 87.5 | 87.7 |
| Mean truncation rate | 0.002 | 0.000 |

Near-identical episode lengths confirm SkyNet isn't "winning" by playing unusually fast or slow — it's genuinely playing better.

### 5.3 Inference-Time Ego Conditioning Ablation

Same trained SkyNet weights, but ego conditioning disabled at inference:

```
SkyNet (full) vs. SkyNet (ego disabled):

Iter 500:  Full 69.0% ██████████████████████████████░░░░░░  (z=12.0, p<10⁻³⁰)
           Ablated 31%

Iter 1000: Full 81.1% █████████████████████████████████████ (z=19.7, p<10⁻⁵⁰)
           Ablated 18.9%
```

This is important: it means the ego conditioning isn't *only* shaping representations during training — the model **actively uses it during planning** at inference time. The benefit isn't just regularization; it's a direct quality signal for MCTS.

### 5.4 Latent Representation Analysis

Linear probes trained on frozen latent vectors test what information the representations encode:

```
R² score (higher = more info encoded):

Feature                    Baseline    SkyNet (pre-cond)    SkyNet (post-cond)
─────────────────────────────────────────────────────────────────────────────
OBSERVABLE features:
  Face-down card count       0.748           0.78                0.845 ★
  Deck size                  0.71            0.74                0.77
  Visible card sum           0.477           0.50                0.539 ★

HIDDEN features:
  Own hidden card sum       -0.21           -0.08               +0.076 ★
  Opponent hidden sum       -0.31           -0.15               +0.168 ★
  True score advantage      -1.062          -0.51               -0.334 ★
─────────────────────────────────────────────────────────────────────────────
  ★ = positive R² on hidden features (baseline can't do this at all)
```

The baseline representation is **blind to hidden information** — linear probes get negative R² on hidden card sums, meaning the latent state encodes *less* than random about what's hidden. SkyNet's ego-conditioned representation crosses into positive territory, encoding weak but real linear traces of hidden-state structure.

---

## 6. The Crossover Effect (Key Insight)

```
Training trajectory of both models:

Phase 1: Data Scarcity (iter 0–250)
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  SkyNet has MORE parameters + auxiliary loss overhead  │
  │  → Gradient bandwidth split between tasks              │
  │  → Core MuZero objective suffers                       │
  │  → SkyNet UNDERPERFORMS baseline                       │
  │                                                        │
  └────────────────────────────────────────────────────────┘

Phase 2: Crossover (iter 250–500)
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  Replay buffer full, opponent pool diverse             │
  │  → Auxiliary heads have enough signal to train well    │
  │  → Winner/rank predictions become accurate             │
  │  → Latent state starts encoding hidden-state structure │
  │                                                        │
  └────────────────────────────────────────────────────────┘

Phase 3: Sustained Advantage (iter 500+)
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  Ego-conditioned representation is now better shaped   │
  │  → MCTS planning operates on richer latent states      │
  │  → Better value + policy estimates from same compute   │
  │  → SkyNet DOMINATES (75.3% win rate, +194 Elo)         │
  │                                                        │
  └────────────────────────────────────────────────────────┘
```

**Practical implication for this project:** Don't judge a belief-augmented model at early training checkpoints. Initial underperformance is expected and doesn't indicate the approach is wrong.

---

## 7. Why Belief Modeling Helps: Two Mechanisms

```
MECHANISM 1: Representation Shaping (training time)
─────────────────────────────────────────────────────

  Without aux heads:              With aux heads (winner/rank):
  Latent state only needs         Latent state must encode enough
  to support value/policy         information to also predict:
  prediction.                       - who will win
                                    - final player rankings
  → Hidden-state info discarded   → Hidden-state info retained
    (no gradient signal to           (needed to predict outcomes)
     encode what you can't see)

  Analogous to: EfficientZero's auxiliary self-supervised losses
  improving sample efficiency by shaping representations.

MECHANISM 2: Perspective Augmentation (training efficiency)
─────────────────────────────────────────────────────────────

  Without ego conditioning:       With ego conditioning:
  1 trajectory = 1 training       1 trajectory = N training
  example (acting player's        examples (one per player
  perspective only)               perspective via augmentation)

  In 4-player Skyjo:              In 4-player Skyjo:
  1 game = 1 signal               1 game = 4 signals
                                  → 4× effective data efficiency
```

---

## 8. Connection to This Project

The recommender system project adapts these ideas from the RL/games domain to collaborative filtering:

| SkyNet (Games) | This Project (Recommenders) |
|---|---|
| Hidden card values | Hidden user preferences |
| Belief over hidden cards | Distribution over user taste vectors |
| Bayes update after each observation | Posterior update after each rating |
| Forgetting factor (not in SkyNet) | Explicit forgetting λ for preference drift |
| Auxiliary heads shape representations | Bayesian uncertainty shapes recommendations |
| Crossover effect: needs enough data | Full-scale Rosie run needed (not smoke test) |
| Ego conditioning = player-specific view | Per-user posterior = user-specific view |
| Non-zero-sum multi-player dynamics | Independent user preference streams |

The key theoretical bridge: **in both settings, maintaining and updating a probabilistic belief over an unobserved quantity leads to better decisions than committing to a point estimate at training time and freezing it.**

SkyNet shows this works empirically in a game domain. This project tests whether the same principle holds in preference modeling under temporal drift.

---

## 9. Limitations and Future Work

### Acknowledged Limitations

```
┌─────────────────────────────────────────────────────────────┐
│  1. IMPLICIT BELIEF                                         │
│     Winner/rank heads predict outcomes, not explicit        │
│     distributions over hidden state. True belief-state      │
│     planners (BetaZero, POMCP) do full Bayesian inference   │
│     over hidden cards — SkyNet only implicitly encodes it.  │
│                                                             │
│  2. SCALE                                                   │
│     Only 2-player Skyjo evaluated. Win signal sparsity      │
│     (1/N chance) and opponent modeling complexity grow       │
│     with player count. Untested at 3–8 players.             │
│                                                             │
│  3. NO HUMAN COMPARISON                                     │
│     Evaluation only against heuristic bots.                 │
│     Relative skill vs. human players unknown.               │
│                                                             │
│  4. ABLATION COVERAGE                                       │
│     Inference-time ablation only. Full training ablations   │
│     (ego conditioning without aux heads, aux heads without  │
│     ego conditioning) not run due to compute constraints.   │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Future Directions

1. **Explicit belief heads** predicting distributions over each face-down card's value, used to weight chance branches in MCTS (Stochastic MuZero style)
2. **Multi-player scaling** experiments (3–8 players) to test where belief-aware advantage breaks down
3. **Asynchronous actor-learner** pipeline to further scale training throughput
4. **Transfer to other imperfect-information games** — Hanabi, Hearts, Coup

---

## 10. Key Takeaways

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. BELIEF-AWARE SUPERVISION WORKS                              │
│     75.3% win rate, +194 Elo, p<10⁻⁵⁰                         │
│     No change to MCTS algorithm needed                          │
│                                                                 │
│  2. DATA FLOW IS THE GATING FACTOR                              │
│     Low throughput → more instability than benefit              │
│     High throughput → consistent, large gains                   │
│     Don't evaluate belief models too early                      │
│                                                                 │
│  3. EGO CONDITIONING IS USED AT INFERENCE, NOT JUST TRAINING    │
│     Ablation: same weights, ego disabled → win rate drops 12%   │
│     The model actively exploits player-specific predictions     │
│     during planning, not just as a training-time regularizer    │
│                                                                 │
│  4. AUXILIARY HEADS SHAPE LATENT REPRESENTATIONS                │
│     Linear probes show ego-conditioned state achieves           │
│     positive R² on hidden features where baseline gets          │
│     negative R² — it encodes what it cannot directly observe    │
│                                                                 │
│  5. THE CROSSOVER EFFECT IS EXPECTED, NOT A FAILURE             │
│     Initial underperformance → data scarcity                    │
│     Patience + sufficient data → decisive advantage             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
