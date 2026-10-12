"""
Compares TRIAGE configurations with their references on the 19-cell grid, paired by cell and seed.

Sources: per-round CSVs written by `scripts/triage_grid.py` (one or more --raw folders) and, for
MNIST, the C0b tables in `results/c0b_reference/` (systems at H ∈ {15, 50, 150}, seeds 72–81).
When a system is in both, the run of this repository is used.

For each candidate and horizon, over the seeds every system involved has run:
  - mean accuracy over cells × seeds;
  - paired difference to each reference (`--refs`), to the best single-signal system and to the
    per-cell oracle choice between the single-signal systems (`--single`); Wilcoxon signed-rank on
    the per-seed means over the 19 cells, and TOST with margin ±`--margin`;
  - per cell: mean difference to the best single signal, and whether it is beyond 2 standard
    errors (win / loss).

Usage:
  venv/bin/python -m scripts.triage_compare --dataset mnist --raw results/<experiment>/raw \\
      --candidates tri_any tri_all --single sr_bin cos_bin --refs fltrust sr_b025 --out results/<experiment>
"""

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [15, 50, 150]
C0B = "results/c0b_reference"
KEYS = ["alpha", "attack_type", "seed"]


def load(dataset: str, raws: List[str]) -> pd.DataFrame:
    frames = []
    if dataset == "mnist" and os.path.exists(f"{C0B}/grade_raw.csv"):
        frames.append(pd.read_csv(f"{C0B}/grade_raw.csv")[["alpha", "attack_type", "seed", "system", "H", "accuracy"]]
                      .assign(source="c0b"))
    for raw in raws:
        files = [f for f in glob.glob(f"{raw}/{dataset}/*/*.csv") if not f.endswith(".log.csv.gz")]
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files])
        d = d[d["round"].isin(HORIZONS)].rename(columns={"round": "H"})
        frames.append(d[["alpha", "attack_type", "seed", "system", "H", "accuracy"]].assign(source="local"))
    d = pd.concat(frames, ignore_index=True)
    d = d.sort_values("source", key=lambda s: s.map({"local": 0, "c0b": 1}))
    return d.drop_duplicates(["alpha", "attack_type", "seed", "system", "H"], keep="first").drop(columns="source")


def tost(diff: np.ndarray, margin: float) -> float:
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return np.nan
    lo = stats.ttest_1samp(diff, -margin, alternative="greater").pvalue
    hi = stats.ttest_1samp(diff, margin, alternative="less").pvalue
    return float(max(lo, hi))


def wilcoxon(diff: np.ndarray) -> float:
    if len(diff) < 2 or np.allclose(diff, 0):
        return np.nan
    return float(stats.wilcoxon(diff).pvalue)


def compare(dataset, raws, candidates, single, refs, margin, out):
    d = load(dataset, raws)
    systems = candidates + single + refs
    lines, rows, cell_rows = [], [], []
    for H in HORIZONS:
        w = d[(d.H == H) & d.system.isin(systems)].pivot_table(index=KEYS, columns="system", values="accuracy")
        for cand in candidates:
            needed = [cand] + single + [r for r in refs if r in w.columns]
            if cand not in w.columns:
                continue
            m = w[needed].dropna()
            if m.empty:
                continue
            seeds = sorted(m.index.get_level_values("seed").unique())
            cell_mean = m.groupby(["alpha", "attack_type"]).mean()
            best_global = cell_mean[single].mean().idxmax()
            per_cell_best = cell_mean[single].idxmax(axis=1)
            oracle = pd.Series([m.loc[i, per_cell_best.loc[i[:2]]] for i in m.index], index=m.index)
            targets = {f"best_single({best_global})": m[best_global], "oracle_single_per_cell": oracle}
            targets.update({r: m[r] for r in needed[1:]})
            for name, ref in targets.items():
                diff = m[cand] - ref
                per_seed = diff.groupby("seed").mean().values
                rows.append({"H": H, "candidate": cand, "reference": name, "n_seeds": len(seeds),
                             "acc_candidate": m[cand].mean(), "acc_reference": ref.mean(),
                             "diff_pp": 100 * diff.mean(), "p_wilcoxon": wilcoxon(per_seed),
                             "p_tost": tost(per_seed, margin)})
            diff = (m[cand] - m[best_global]).groupby(["alpha", "attack_type"])
            for cell, g in diff:
                se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
                verdict = "win" if g.mean() > 2 * se else "loss" if g.mean() < -2 * se else "tie"
                cell_rows.append({"H": H, "candidate": cand, "alpha": cell[0], "attack_type": cell[1],
                                  "diff_pp": 100 * g.mean(), "se_pp": 100 * se, "verdict": verdict})
    summary, cells = pd.DataFrame(rows), pd.DataFrame(cell_rows)
    os.makedirs(out, exist_ok=True)
    summary.to_csv(f"{out}/compare_{dataset}.csv", index=False)
    cells.to_csv(f"{out}/compare_{dataset}_cells.csv", index=False)
    pd.set_option("display.width", 250)
    lines.append(f"=== {dataset}: candidates vs references (diff in p.p., paired by cell and seed) ===")
    lines.append(summary.round(4).to_string(index=False))
    if not cells.empty:
        h = cells[cells.H == max(cells.H)]
        lines.append(f"\n=== {dataset}: per cell vs the best single signal, H = {max(cells.H)} ===")
        lines.append(h.pivot_table(index=["alpha", "attack_type"], columns="candidate", values="diff_pp")
                     .round(2).to_string())
        lines.append("\nwins / ties / losses (beyond 2 SE):")
        lines.append(h.groupby("candidate")["verdict"].value_counts().unstack(fill_value=0).to_string())
    text = "\n".join(lines)
    with open(f"{out}/compare_{dataset}.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--raw", nargs="+", default=[])
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--single", nargs="+", required=True)
    p.add_argument("--refs", nargs="*", default=[])
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    compare(a.dataset, a.raw, a.candidates, a.single, a.refs, a.margin, a.out)


if __name__ == "__main__":
    main()
