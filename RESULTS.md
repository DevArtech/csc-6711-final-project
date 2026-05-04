# Experiment Results — Belief-Aware Recommender Systems

**Dataset:** MovieLens-1M · 6,040 users · 3,706 movies · 1,000,209 ratings  
**Hardware:** MSOE Rosie V100 GPU  
**Split:** 60% warm (train) / 30% stream / 10% test — chronological per user  
**Drift users:** 1,000 users with synthetically swapped stream histories  
**Evaluation:** 100 negative samples per test rating, rank the true item among 101 candidates  

---

## 1. Metrics Explained

### HR@K — Hit Rate at K (ranking metric, higher is better)
Did the movie the user actually watched appear in the top K recommendations?

```
For each test interaction:
  → Sample 100 random movies the user has NOT seen (negatives)
  → Rank the true movie among those 101 candidates
  → HR@10 = 1 if true movie lands in top 10, else 0
  → Average over all test interactions
```

This is the primary metric. A score of 0.535 means the right movie appeared in the top 10 roughly 53.5% of the time.

### NDCG@K — Normalized Discounted Cumulative Gain (ranking metric, higher is better)
Same as HR@K but rewards the model more when the true movie lands closer to the top of the list. Rank #1 is worth much more than rank #10.

```
NDCG@10 = 1 / log2(rank + 1)   if rank ≤ 10, else 0
```

### RMSE — Root Mean Squared Error (rating prediction metric, lower is better)
How far off is the model's predicted star rating from the actual star rating?

```
RMSE = sqrt( mean( (predicted_rating - actual_rating)² ) )
```

### Which metric matters most?
**Ranking (HR@10, NDCG@10) is what matters for a real system.** Nobody sees predicted star ratings — they see a ranked list of movies. RMSE is useful for understanding rating calibration but is a poor proxy for recommendation quality. Our results demonstrate this gap clearly: the models with the best RMSE (Static MF MSE, Bayesian MF) are beaten on ranking by a simple popularity baseline.

Note: Static MF (BPR) and Sequential are trained with ranking losses — their RMSE values exist but are not meaningful for comparison.

---

## 2. Final Rankings (Full Test Set — 102,759 ratings)

![Ranking Comparison](../recsys/runs/plots/ranking_comparison_original.png)

### Full Table

| Model | HR@5 | HR@10 | HR@20 | NDCG@10 | RMSE | MAE |
|---|---|---|---|---|---|---|
| Sequential | 0.403 | **0.535** | 0.680 | **0.328** | — | — |
| **Static MF (BPR)** | 0.287 | **0.453** | 0.661 | 0.239 | — | — |
| Popularity | 0.262 | 0.411 | 0.612 | 0.218 | 1.158 | 0.959 |
| Static MF (MSE) | 0.158 | 0.250 | 0.380 | 0.133 | **0.926** | **0.731** |
| Bayesian MF | 0.138 | 0.222 | 0.345 | 0.117 | 1.023 | 0.801 |

**Key result:** Static MF with BPR loss jumps from HR@10 = 0.250 to **0.453**, surpassing the popularity baseline (0.411). Switching the training objective from rating prediction (MSE) to ranking (BPR) was the fix.

---

## 3. Drift Users (17,811 ratings from 1,000 preference-shifted users)

These users had their stream histories swapped halfway through — the scenario
the Bayesian model was specifically designed to handle.

![Drift Comparison](../recsys/runs/plots/drift_comparison_original.png)

| Model | HR@10 (drift) | NDCG@10 (drift) | RMSE (drift) |
|---|---|---|---|
| Sequential | **0.535** | **0.325** | — |
| Static MF (BPR) | 0.442 | 0.235 | — |
| Popularity | 0.404 | 0.216 | 1.167 |
| Static MF (MSE) | 0.247 | 0.132 | 0.938 |
| Bayesian MF | 0.217 | 0.113 | 1.041 |

---

## 4. Streaming Curves — Performance Over Time

How HR@10 evolved as each model processed the 299,708 stream interactions.

![Streaming Curves](../recsys/runs/plots/streaming_curves.png)

| Step | Sequential | Static MF | Bayesian |
|---|---|---|---|
| 20K | 0.322 | 0.243 | 0.241 |
| 60K | 0.351 | 0.244 | 0.238 |
| 100K | 0.387 | 0.245 | 0.235 |
| 140K | 0.413 | 0.247 | 0.231 |
| 180K | 0.448 | 0.247 | 0.229 |
| 220K | 0.475 | 0.249 | 0.225 |
| 260K | 0.509 | 0.250 | 0.223 |
| 299K | 0.535 | 0.251 | 0.221 |

---

## 5. Rating Prediction Error

![Rating Error](../recsys/runs/plots/rating_error.png)

---

## 6. BPR vs MSE — Effect of Training Objective

![BPR vs MSE](../recsys/runs/plots/bpr_vs_mse.png)

---

## 7. Bayesian Hyperparameter Tuning

![Bayesian Tuning](../recsys/runs/plots/bayesian_tuning.png)



We tested whether tuning the forgetting factor (which controls how fast old beliefs decay) could improve Bayesian MF's performance on drift users. All 4 configs performed **worse** than the baseline (forgetting=1.0), with more aggressive forgetting causing larger degradation.

| Config | HR@10 (overall) | NDCG@10 | RMSE | HR@10 (drift) |
|---|---|---|---|---|
| **f=1.00 (baseline)** | **0.222** | **0.117** | **1.023** | **0.217** |
| f=0.98 | 0.209 | 0.110 | 1.129 | 0.199 |
| f=0.95 | 0.199 | 0.104 | 1.418 | 0.192 |
| f=0.95, nv=0.25 | 0.190 | 0.098 | 1.479 | 0.187 |
| f=0.90 | 0.188 | 0.098 | 1.959 | 0.182 |

**Interpretation:** Forgetting=1.0 is actually optimal for this model and dataset. The Bayesian model's problem is not the forgetting factor — it's that the model was never trained with a ranking objective. With more forgetting, the model loses useful accumulated signal faster than it gains from recency weighting.

---

## 8. Discussion

### BPR fixes Static MF's ranking problem
Replacing MSE loss with BPR (Bayesian Personalized Ranking) caused Static MF to jump from HR@10 = 0.250 to 0.453 — a **81% improvement** and above the popularity baseline (0.411). This confirms the hypothesis: Static MF was underperforming not because matrix factorization is wrong for this task, but because MSE loss trains for rating prediction rather than ranking. BPR directly trains the model to score a watched item above a random unwatched item, which is exactly what HR@10 measures.

### Sequential dominates overall
Sequential outperforms every model on HR@10 (0.535) and NDCG@10 (0.328). The GRU reads full watch history in chronological order — more stream history means richer context, explaining the steady climb in the streaming curve. Its training objective (predict the next item) is naturally aligned with ranking evaluation.

### Popularity beats MSE-trained models — but not BPR
The original run showed popularity (0.411) beating both Static MF MSE (0.250) and Bayesian MF (0.222) on ranking. After switching Static MF to BPR, the learned model now surpasses popularity. This validates the diagnosis: the issue was always the training objective, not the model architecture.

### Bayesian MF's forgetting factor is not the issue
All tested forgetting values (0.90, 0.95, 0.98) made performance worse than the default (1.0). The Bayesian model's fundamental limitation is that it optimizes for rating prediction via its Bayesian update rule — the residual `(actual_rating - predicted_rating)` is baked into the update math and cannot be replaced with a ranking signal without redesigning the model entirely. Tuning forgetting cannot fix a training objective mismatch.

### Static MF (MSE) wins on RMSE
Static MF (MSE) achieves the best RMSE (0.926) because it's directly trained to minimize rating prediction error. This is the only metric where the MSE-trained models excel, confirming that RMSE and ranking are measuring fundamentally different things.

---

## 9. Success Criteria Check

| Criterion | Result | Met? |
|---|---|---|
| Bayesian wins on NDCG@10 and HR@10 overall | Lost to Sequential, BPR, and Popularity | ❌ |
| Bayesian ≤ 1–2% worse than best on RMSE | 10.5% worse than Static MF MSE | ❌ |
| Bayesian clear ranking advantage on drift users | No improvement on drift vs overall | ❌ |
| BPR Static MF beats popularity baseline | HR@10 0.453 vs 0.411 ✓ | ✅ |

The core hypothesis — that belief-aware online updates improve recommendations under preference drift — was not confirmed. However, the BPR experiment confirmed a complementary finding: the training objective (not model capacity) was the primary bottleneck for Static MF.

---

## 10. Summary Table — All Models

| Model | HR@10 | NDCG@10 | RMSE | Training Objective |
|---|---|---|---|---|
| Sequential | **0.535** | **0.328** | — | Next-item ranking |
| Static MF (BPR) | 0.453 | 0.239 | — | BPR ranking loss |
| Popularity | 0.411 | 0.218 | 1.158 | None |
| Static MF (MSE) | 0.250 | 0.133 | **0.926** | MSE rating prediction |
| Bayesian MF (f=1.0) | 0.222 | 0.117 | 1.023 | MSE + online Bayes update |
