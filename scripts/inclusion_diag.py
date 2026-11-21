"""
Summarizes the per-client logs (`--log` in `scripts/triage_grid.py`) of one cell over windows of
rounds: how often the Byzantine and the honest clients are included and flagged by each signal,
and the ratio between the mean update norm of the Byzantine and of the honest clients.

Usage:
  venv/bin/python -m scripts.inclusion_diag --logs "results/collapse_diag/raw/bloodmnist/tri_all/*.log.csv.gz" \\
      --out results/collapse_diag/inclusion.txt
"""

import argparse
import glob

import pandas as pd

WINDOWS = [0, 50, 100, 150, 200, 250, 300]


def summarize(pattern: str, windows=WINDOWS) -> str:
    frames = []
    for f in sorted(glob.glob(pattern)):
        frames.append(pd.read_csv(f).assign(source=f))
    d = pd.concat(frames)
    d["norm"] = -d["neg_norm"]
    d["rounds"] = pd.cut(d["round"], windows)
    flags = [c for c in d.columns if c.endswith("_flag")]
    by_group = d.groupby(["rounds", "is_byz"], observed=True)
    table = by_group[["included"] + flags].mean().unstack("is_byz")
    norm = by_group["norm"].mean().unstack("is_byz")
    table[("norm_ratio", "byz/honest")] = norm[True] / norm[False]
    pd.set_option("display.width", 200)
    return f"{len(frames)} runs\n" + table.round(3).to_string()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True, help="glob of .log.csv.gz files")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    text = summarize(a.logs)
    with open(a.out, "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
