"""Print a comparison table of all Bayesian tuning runs vs the original baseline."""
from __future__ import annotations

import json
from pathlib import Path


CONFIGS = [
    ("baseline (f=1.00, nv=0.50)", "runs/compare"),
    ("f090_nv050 (f=0.90, nv=0.50)", "runs/compare_tune/f090_nv050"),
    ("f095_nv050 (f=0.95, nv=0.50)", "runs/compare_tune/f095_nv050"),
    ("f098_nv050 (f=0.98, nv=0.50)", "runs/compare_tune/f098_nv050"),
    ("f095_nv025 (f=0.95, nv=0.25)", "runs/compare_tune/f095_nv025"),
]

BASE = Path(__file__).resolve().parent.parent / "recsys"


def load(path: Path, key: str = "bayesian_mf") -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get(key)


def fmt(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "—"


def main() -> None:
    header = f"{'Config':<40} {'HR@10':>7} {'NDCG@10':>8} {'RMSE':>7} | {'HR@10(drift)':>13} {'NDCG@10(drift)':>15}"
    print(header)
    print("-" * len(header))

    best_hr = 0.0
    rows = []

    for label, rel_dir in CONFIGS:
        d = BASE / rel_dir
        overall = load(d / "summary.json")
        drift = load(d / "drift_subset_summary.json")

        hr10  = overall["hr@10"]  if overall else None
        nd10  = overall["ndcg@10"] if overall else None
        rmse  = overall["rmse"]   if overall else None
        dhr   = drift["hr@10"]    if drift   else None
        dnd   = drift["ndcg@10"]  if drift   else None

        if hr10 and hr10 > best_hr:
            best_hr = hr10

        rows.append((label, hr10, nd10, rmse, dhr, dnd))

    for label, hr10, nd10, rmse, dhr, dnd in rows:
        marker = " *" if hr10 and abs(hr10 - best_hr) < 1e-6 else "  "
        print(f"{label:<40} {fmt(hr10):>7} {fmt(nd10):>8} {fmt(rmse):>7} | {fmt(dhr):>13} {fmt(dnd):>15}{marker}")

    print()
    print("* = best HR@10")
    print("Drift users = 1,000 users with preferences swapped mid-stream")


if __name__ == "__main__":
    main()
