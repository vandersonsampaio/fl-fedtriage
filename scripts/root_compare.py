"""
Compares systems run with a modified root dataset (`<system>@<root_tag>`, see `scripts/triage_grid.py`)
with the same system on the default root, paired by cell and seed, at H = 150.

For each dataset, system and root variant present in the raw folders:
  - mean accuracy with the default root and with the variant, and their difference;
  - Wilcoxon signed-rank on the per-seed means over the 19 cells and number of seeds where the
    variant is ahead;
  - the worst cell (largest mean loss).
For each variant, `--candidate` is also compared with every other system on the same root.

Usage:
  venv/bin/python -m scripts.root_compare --dataset bloodmnist \\
      --raw results/bloodmnist_reference/raw results/root_cost/raw --out results/root_cost
"""

import argparse
import os

import pandas as pd

from scripts.triage_compare import KEYS, load, wilcoxon

H = 150


def paired(w: pd.DataFrame, a: str, b: str) -> dict:
    m = w[[a, b]].dropna()
    diff = m[a] - m[b]
    per_seed = diff.groupby("seed").mean()
    cell = diff.groupby(["alpha", "attack_type"]).mean()
    worst = cell.idxmin()
    return {"n": len(m), "n_seeds": len(per_seed), "acc_a": m[a].mean(), "acc_b": m[b].mean(),
            "diff_pp": 100 * diff.mean(), "p_wilcoxon": wilcoxon(per_seed.values),
            "seeds_ahead": int((per_seed > 0).sum()), "worst_cell": f"a{worst[0]} {worst[1]}",
            "worst_cell_pp": 100 * cell.min()}


def compare(dataset, raws, candidate, seeds, out):
    d = load(dataset, raws)
    d = d[(d.H == H) & d.seed.isin(seeds)]
    w = d.pivot_table(index=KEYS, columns="system", values="accuracy")
    variants = sorted({c.split("@", 1)[1] for c in w.columns if "@" in c})
    bases = sorted({c.split("@", 1)[0] for c in w.columns if "@" in c})
    vs_default, vs_same_root = [], []
    for v in variants:
        for s in bases:
            if f"{s}@{v}" in w and s in w:
                vs_default.append({"system": s, "root": v, **paired(w, f"{s}@{v}", s)})
        for tag in ["", f"@{v}"]:
            if f"{candidate}{tag}" not in w:
                continue
            for s in bases:
                if s != candidate and f"{s}{tag}" in w:
                    vs_same_root.append({"root": tag[1:] or "default", "reference": s,
                                         **paired(w, f"{candidate}{tag}", f"{s}{tag}")})
    a = pd.DataFrame(vs_default)
    b = pd.DataFrame(vs_same_root).drop_duplicates(["root", "reference"])
    os.makedirs(out, exist_ok=True)
    a.to_csv(f"{out}/root_vs_default_{dataset}.csv", index=False)
    b.to_csv(f"{out}/root_{candidate}_vs_same_root_{dataset}.csv", index=False)
    pd.set_option("display.width", 250)
    text = "\n".join([
        f"=== {dataset}: variant root vs default root, same system (p.p., H = {H}) ===",
        a.round(4).to_string(index=False) if not a.empty else "(none)",
        f"\n=== {dataset}: {candidate} vs the other systems on the same root (p.p., H = {H}) ===",
        b.round(4).to_string(index=False) if not b.empty else "(none)",
    ])
    with open(f"{out}/root_compare_{dataset}.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument("--candidate", default="tri_all")
    p.add_argument("--seeds", nargs="+", type=int, default=[72, 73, 74, 75, 76])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    compare(a.dataset, a.raw, a.candidate, a.seeds, a.out)


if __name__ == "__main__":
    main()
