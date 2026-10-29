#!/usr/bin/env python3
"""One-time data setup: extract data/dataset_to_release.zip -> data/dataset/.

The raw CSVs (~320 MB uncompressed) are too large to push to GitHub, so the
repository ships only the compressed archive (data/dataset_to_release.zip,
~73 MB) plus this script. Run once after cloning:

    python setup_data.py

The script is idempotent: if data/dataset/ already contains the four
expected files, it does nothing (and says so). Standard library only
(zipfile, os, sys) — no third-party dependencies.

After extraction, the full pipeline is available: open run.ipynb in
Jupyter and Run All.
"""

import os
import shutil
import sys
import zipfile

# the four files the pipeline needs; anything else in the archive is skipped
EXPECTED = {
    "x_train.csv": 200_000_000,     # minimum plausible sizes (bytes), sanity floor
    "y_train.csv": 3_000_000,
    "x_test.csv": 50_000_000,
    "sample_submission.csv": 500_000,
}


def repo_root():
    """Directory containing this script (works from any cwd)."""
    return os.path.dirname(os.path.abspath(__file__))


def dataset_ready(dest):
    """True if all expected files exist and exceed their sanity floor."""
    for name, min_size in EXPECTED.items():
        p = os.path.join(dest, name)
        if not os.path.isfile(p) or os.path.getsize(p) < min_size:
            return False
    return True


def extract(zip_path, dest, verbose=True):
    """Extract the expected CSVs from the archive into dest (flat)."""
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for name, min_size in EXPECTED.items():
            member = [m for m in names if m.endswith("/" + name) or m == name]
            if not member:
                raise FileNotFoundError(
                    f"{name!r} not found inside {zip_path}")
            # endswith("/x.csv") guards against e.g. y_train.csv matching
            # x_train.csv: compare on the basename, exactly one hit expected
            member = [m for m in member
                      if os.path.basename(m) == name]
            if len(member) != 1:
                raise ValueError(f"expected exactly one {name!r} in archive, "
                                 f"found {member}")
            target = os.path.join(dest, name)
            with z.open(member[0]) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            size = os.path.getsize(target)
            if size < min_size:
                raise ValueError(f"{target} is only {size} bytes "
                                 f"(expected >= {min_size}) - corrupt archive?")
            if verbose:
                print(f"  extracted {name:22s} {size:>12,} bytes")
    if not dataset_ready(dest):
        raise RuntimeError("extraction finished but validation failed")
    return True


def main():
    root = repo_root()
    zip_path = os.path.join(root, "data", "dataset_to_release.zip")
    dest = os.path.join(root, "data", "dataset")

    if not os.path.isfile(zip_path):
        sys.exit(f"ERROR: archive not found: {zip_path}")

    if dataset_ready(dest):
        print("data/dataset/ already present and complete - nothing to do.")
        return 0

    print(f"extracting {os.path.relpath(zip_path, root)} "
          f"-> {os.path.relpath(dest, root)}/ ...")
    extract(zip_path, dest)
    print("done. Data is ready - open run.ipynb and Run All.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
