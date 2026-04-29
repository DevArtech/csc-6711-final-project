from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_ml1m(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    zip_path = output_root / "ml-1m.zip"
    extract_root = output_root / "ml-1m"

    if extract_root.exists() and any(extract_root.iterdir()):
        print(f"Dataset already present at {extract_root}")
        return extract_root

    print(f"Downloading MovieLens-1M from {ML1M_URL}")
    with urllib.request.urlopen(ML1M_URL) as response, zip_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)

    print(f"Extracting archive to {extract_root}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_root)

    extracted_dir = output_root / "ml-1m"
    if not extracted_dir.exists():
        nested = output_root / "ml-1m" / "ml-1m"
        if nested.exists():
            extracted_dir = nested
        else:
            raise RuntimeError("Could not find extracted ml-1m directory.")

    print(f"Done. Dataset available at {extracted_dir}")
    return extracted_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MovieLens-1M dataset.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("recsys/data/raw"),
        help="Directory where ml-1m.zip and extracted files are stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_ml1m(args.output_root)


if __name__ == "__main__":
    main()
