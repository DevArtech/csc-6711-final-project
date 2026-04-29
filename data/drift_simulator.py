from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_interactions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_stream_by_user(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    stream_by_user: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "stream":
            stream_by_user[int(row["user_id"])].append(row)
    for seq in stream_by_user.values():
        seq.sort(key=lambda r: (int(r["timestamp"]), int(r["item_id"])))
    return stream_by_user


def create_drift_stream(
    interactions_csv: Path,
    output_csv: Path,
    output_meta: Path,
    num_users: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    rows = load_interactions(interactions_csv)
    stream_by_user = build_stream_by_user(rows)

    # Keep threshold low so smoke-test sized datasets can still build drift cohorts.
    candidates = [u for u, seq in stream_by_user.items() if len(seq) >= 2]
    rng.shuffle(candidates)
    selected = candidates[: min(num_users, len(candidates))]
    if len(selected) < 2:
        # Graceful fallback for tiny or heavily filtered datasets:
        # write a pass-through copy and empty drift cohort metadata.
        out_rows = sorted(rows, key=lambda r: (int(r["timestamp"]), int(r["user_id"]), int(r["item_id"])))
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["user_id", "item_id", "rating", "timestamp", "split", "user_order_idx"],
            )
            writer.writeheader()
            writer.writerows(out_rows)

        meta = {
            "seed": seed,
            "requested_users": num_users,
            "actual_drift_users": [],
            "swap_pairs": [],
            "output_csv": str(output_csv),
            "note": "Insufficient users with stream interactions; emitted pass-through stream.",
        }
        with output_meta.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        print(json.dumps(meta, indent=2, sort_keys=True))
        return

    drift_users = set(selected)
    paired = selected[:]
    if len(paired) % 2 == 1:
        paired = paired[:-1]
    rng.shuffle(paired)

    swap_pairs = [(paired[i], paired[i + 1]) for i in range(0, len(paired), 2)]
    partner_map = {a: b for a, b in swap_pairs}
    partner_map.update({b: a for a, b in swap_pairs})

    out_rows: list[dict[str, str]] = []
    for row in rows:
        user = int(row["user_id"])
        if row["split"] != "stream" or user not in drift_users or user not in partner_map:
            out_rows.append(row)
            continue

        seq = stream_by_user[user]
        half = len(seq) // 2
        row_rank = seq.index(row)
        if row_rank < half:
            out_rows.append(row)
            continue

        partner = partner_map[user]
        partner_seq = stream_by_user[partner]
        partner_half = len(partner_seq) // 2
        partner_idx = min(partner_half + (row_rank - half), len(partner_seq) - 1)
        swapped = dict(row)
        swapped["item_id"] = partner_seq[partner_idx]["item_id"]
        swapped["rating"] = partner_seq[partner_idx]["rating"]
        out_rows.append(swapped)

    out_rows.sort(key=lambda r: (int(r["timestamp"]), int(r["user_id"]), int(r["item_id"])))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "item_id", "rating", "timestamp", "split", "user_order_idx"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    meta = {
        "seed": seed,
        "requested_users": num_users,
        "actual_drift_users": sorted(drift_users),
        "swap_pairs": swap_pairs,
        "output_csv": str(output_csv),
    }
    with output_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic drift stream by swapping user second-halves.")
    parser.add_argument(
        "--interactions-csv",
        type=Path,
        default=Path("recsys/data/processed/interactions.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("recsys/data/processed/interactions_drift.csv"),
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=Path("recsys/data/processed/drift_meta.json"),
    )
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_drift_stream(
        interactions_csv=args.interactions_csv,
        output_csv=args.output_csv,
        output_meta=args.output_meta,
        num_users=args.num_users,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
