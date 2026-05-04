# Bayesian MF Hyperparameter Tuning

## What we're doing
Re-run Bayesian MF with different forgetting factors (and optionally noise_var) to fix the core underperformance identified in the first run. The model got worse over the stream because forgetting=1.0 means old ratings never decay — we want to test values where recent behavior actually dominates.

## Why
The streaming curve showed Bayesian declining from HR@10=0.241 → 0.221 over 300K steps. With forgetting=1.0, every past rating has equal weight forever. At forgetting=0.95, a rating 20 steps ago has only 36% of its original weight, so preference drift can actually take effect.

## Plan

1. Add a new sbatch template `recsys_bayesian_tune.sbatch` that accepts FORGETTING and NOISE_VAR as env vars and passes them as CLI args to train_bayesian_mf.py.

2. Add a submission script `submit_bayesian_tuning.sh` that launches one job per config:
   - forgetting=0.95, noise_var=0.5  (strong decay, current noise)
   - forgetting=0.98, noise_var=0.5  (mild decay, current noise)
   - forgetting=0.95, noise_var=0.25 (strong decay, tighter noise)
   - Each job gets its own output dir: runs/bayesian_tune/f095_nv05, etc.

3. After all 3 training jobs finish, run compare_recommenders for each checkpoint against the same test set — or add a lightweight eval-only script that just loads a checkpoint and runs the drift subset eval (faster than full compare).

4. Collect results into a tuning summary table comparing all configs side by side on HR@10, NDCG@10 (overall + drift subset).
