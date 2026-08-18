"""Validate and extract the preprocessed IMDB arrays bundled with the repo."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


EXPECTED_SHA256 = "df6be1272b0a773fb7a89c5f158fa3c96bd425b7ba4f882434ea0a3b3937e819"
EXPECTED_FILES = {"x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    archive = data_dir / "imdb_preprocessed_data.zip"

    if not archive.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")

    actual_hash = sha256(archive)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"Dataset checksum mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
        )

    with zipfile.ZipFile(archive) as bundle:
        archive_files = {Path(name).name for name in bundle.namelist()}
        if archive_files != EXPECTED_FILES:
            raise ValueError(
                f"Unexpected archive contents: {sorted(archive_files)}"
            )
        bundle.extractall(data_dir)

    print(f"Extracted {len(EXPECTED_FILES)} files to {data_dir}")


if __name__ == "__main__":
    main()
