from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_movies(movies_path: Path) -> tuple[dict[str, list[str]], list[str]]:
    item_to_genres: dict[str, list[str]] = {}
    genre_vocab: set[str] = set()
    with movies_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 3:
                continue
            raw_item, _title, genres = parts
            genre_list = [g for g in genres.split("|") if g]
            item_to_genres[raw_item] = genre_list
            genre_vocab.update(genre_list)
    return item_to_genres, sorted(genre_vocab)


def read_ratings(ratings_path: Path) -> list[tuple[str, str, float, int]]:
    rows: list[tuple[str, str, float, int]] = []
    with ratings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 4:
                continue
            raw_user, raw_item, rating_str, ts_str = parts
            rows.append((raw_user, raw_item, float(rating_str), int(ts_str)))
    return rows


def split_user_history(history: list[tuple[int, int, float, int]]) -> list[str]:
    n = len(history)
    warm_end = max(1, int(n * 0.6))
    stream_end = max(warm_end + 1, int(n * 0.9))
    stream_end = min(stream_end, n - 1)
    labels = []
    for idx in range(n):
        if idx < warm_end:
            labels.append("warm")
        elif idx < stream_end:
            labels.append("stream")
        else:
            labels.append("test")
    return labels


def preprocess(
    input_dir: Path,
    output_dir: Path,
    min_interactions: int,
    max_users: int | None,
) -> None:
    ratings_path = input_dir / "ratings.dat"
    movies_path = input_dir / "movies.dat"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"Expected ratings.dat and movies.dat in {input_dir}. "
            "Run recsys/data/download_ml1m.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    item_to_genres_raw, genre_names = read_movies(movies_path)
    ratings = read_ratings(ratings_path)
    ratings.sort(key=lambda x: (x[0], x[3], x[1]))

    per_user_raw: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for raw_user, raw_item, rating, ts in ratings:
        per_user_raw[raw_user].append((raw_item, rating, ts))

    candidate_users = [u for u, seq in per_user_raw.items() if len(seq) >= min_interactions]
    candidate_users.sort(key=lambda u: (-len(per_user_raw[u]), u))
    if max_users is not None:
        candidate_users = candidate_users[:max_users]

    user_id_map = {raw_user: idx for idx, raw_user in enumerate(candidate_users)}
    used_items_raw: set[str] = set()
    for raw_user in candidate_users:
        for raw_item, _rating, _ts in per_user_raw[raw_user]:
            used_items_raw.add(raw_item)
    item_id_map = {raw_item: idx for idx, raw_item in enumerate(sorted(used_items_raw, key=int))}

    genre_to_id = {name: idx for idx, name in enumerate(genre_names)}
    item_to_genre_ids: dict[int, list[int]] = {}
    for raw_item, item_id in item_id_map.items():
        genre_list = item_to_genres_raw.get(raw_item, [])
        item_to_genre_ids[item_id] = [genre_to_id[g] for g in genre_list if g in genre_to_id]

    rows_out: list[dict[str, str | int | float]] = []
    warm_count = 0
    stream_count = 0
    test_count = 0
    rating_sum = 0.0
    rating_count = 0

    for raw_user in candidate_users:
        user_id = user_id_map[raw_user]
        user_hist = [
            (user_id, item_id_map[raw_item], rating, ts)
            for raw_item, rating, ts in sorted(per_user_raw[raw_user], key=lambda x: (x[2], x[0]))
            if raw_item in item_id_map
        ]
        split_labels = split_user_history(user_hist)

        for order_idx, ((uid, iid, rating, ts), split) in enumerate(zip(user_hist, split_labels)):
            rows_out.append(
                {
                    "user_id": uid,
                    "item_id": iid,
                    "rating": rating,
                    "timestamp": ts,
                    "split": split,
                    "user_order_idx": order_idx,
                }
            )
            rating_sum += rating
            rating_count += 1
            if split == "warm":
                warm_count += 1
            elif split == "stream":
                stream_count += 1
            else:
                test_count += 1

    rows_out.sort(key=lambda r: (int(r["user_id"]), int(r["timestamp"]), int(r["item_id"])))

    interactions_csv = output_dir / "interactions.csv"
    with interactions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "item_id", "rating", "timestamp", "split", "user_order_idx"],
        )
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    with (output_dir / "user_id_map.json").open("w", encoding="utf-8") as handle:
        json.dump(user_id_map, handle, indent=2, sort_keys=True)
    with (output_dir / "item_id_map.json").open("w", encoding="utf-8") as handle:
        json.dump(item_id_map, handle, indent=2, sort_keys=True)

    with (output_dir / "item_genres.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "genre_to_id": genre_to_id,
                "genre_names": genre_names,
                "item_to_genres": item_to_genre_ids,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    metadata = {
        "num_users": len(user_id_map),
        "num_items": len(item_id_map),
        "num_interactions": len(rows_out),
        "warm_interactions": warm_count,
        "stream_interactions": stream_count,
        "test_interactions": test_count,
        "global_mean_rating": (rating_sum / max(1, rating_count)),
        "min_interactions_filter": min_interactions,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote interactions to {interactions_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens-1M into chronological splits.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("recsys/data/raw/ml-1m"),
        help="Directory containing ratings.dat and movies.dat",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recsys/data/processed"),
        help="Directory for processed outputs",
    )
    parser.add_argument("--min-interactions", type=int, default=20, help="Filter users by min interactions.")
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap on users for quick experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_interactions=args.min_interactions,
        max_users=args.max_users,
    )


if __name__ == "__main__":
    main()
