"""
Compares B2.6 runs re-executed here with the CSVs published in fl-gradf
(`results/b26_decomposicao/raw/`). Criterion: |Δ| = 0 on every cell.

Usage:
  venv/bin/python -m scripts.b26_decomposicao run --variant sr_bin --seed 52   # writes results/b26_decomposicao/raw/
  venv/bin/python -m scripts.parity_fl_gradf --gradf ../fl-fedmed --new results/b26_decomposicao/raw sr_bin_seed52.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

KEYS = ["alpha", "attack_type", "variant", "seed"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gradf", required=True, help="local clone of fl-gradf")
    p.add_argument("--new", default="results/b26_decomposicao/raw")
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    ok = True
    for name in a.files:
        old = pd.read_csv(os.path.join(a.gradf, "results/b26_decomposicao/raw", name))
        new = pd.read_csv(os.path.join(a.new, name))
        m = old.merge(new, on=KEYS, suffixes=("_old", "_new"))
        d = np.abs(m["accuracy_old"] - m["accuracy_new"])
        good = len(m) == len(old) == len(new) and float(d.max()) == 0.0
        ok &= good
        print(f"{name}: {len(old)} vs {len(new)} rows, matched {len(m)}, max|Δ| = {d.max():.2e}, "
              f"cells with Δ≠0: {(d > 0).sum()}")
    print("PARITY", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
