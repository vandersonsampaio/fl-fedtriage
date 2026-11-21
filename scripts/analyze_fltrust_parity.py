"""
FLTrust with step parity against native FLTrust and native tri_all, on seeds 72-76.

Per seed, mean over the 19 cells, CI95 with t. Reading rule and inputs in
results/fltrust_parity/ADDENDUM.md. Writes results/fltrust_parity/analysis/.

Usage: venv/bin/python scripts/analyze_fltrust_parity.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/fltrust_parity"
SEEDS = [72, 73, 74, 75, 76]
KEY = ["alpha", "attack_type", "seed"]
# dataset, root, horizon, native FLTrust folder, native tri_all folder
BLOOD, MNIST = "bloodmnist", "mnist"
CASES = [
    (BLOOD, 100, 250, "results/horizon300/raw/bloodmnist/fltrust", "results/horizon300/raw/bloodmnist/tri_all"),
    (BLOOD, 100, 150, "results/horizon300/raw/bloodmnist/fltrust", "results/horizon300/raw/bloodmnist/tri_all"),
    (BLOOD, 20, 150, "results/root_cost/raw/bloodmnist/fltrust@r20", "results/root_cost/raw/bloodmnist/tri_all@r20"),
    (MNIST, 100, 150, None, "results/f2_agreement/raw/mnist/tri_all"),
    (MNIST, 20, 150, "results/root_cost/raw/mnist/fltrust@r20", "results/root_cost/raw/mnist/tri_all@r20"),
]


def at_horizon(folder, H):
    frames = [pd.read_csv(f) for s in SEEDS for f in glob.glob(f"{folder}/*_seed{s}.csv")]
    d = pd.concat(frames, ignore_index=True)
    d = d[d["round"] == H]
    if len(d) != 19 * len(SEEDS):
        raise SystemExit(f"{folder}: {len(d)} rows at H={H}, expected {19 * len(SEEDS)}")
    return d.set_index(KEY).accuracy


def mnist_native_fltrust(H):
    d = pd.read_csv("results/c0b_reference/grade_raw.csv")
    d = d[(d.system == "fltrust") & (d.H == H) & d.seed.isin(SEEDS)]
    if len(d) != 19 * len(SEEDS):
        raise SystemExit(f"grade_raw: {len(d)} rows at H={H}")
    return d.set_index(KEY).accuracy


def ci(diff):
    x = diff.groupby(level="seed").mean().values
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.975, len(x) - 1)
    return m, m - q * se, m + q * se, x


def main():
    os.makedirs(f"{OUT}/analysis", exist_ok=True)
    rows, cells, out = [], [], []
    for dataset, root, H, nat, tri in CASES:
        tag = "fltrust@stmedian" if root == 100 else "fltrust@r20@stmedian"
        par = at_horizon(f"{OUT}/raw/{dataset}/{tag}", H)
        fl = mnist_native_fltrust(H) if nat is None else at_horizon(nat, H)
        ta = at_horizon(tri, H)
        res = {}
        out.append(f"## {dataset}, root {root}, H = {H}")
        out.append(f"Means: FLTrust native {100 * fl.mean():.2f}, FLTrust parity {100 * par.mean():.2f}, tri_all native {100 * ta.mean():.2f}")
        for name, d in (("fltrust_parity_minus_native", par - fl), ("tri_all_minus_fltrust_native", ta - fl),
                        ("tri_all_minus_fltrust_parity", ta - par)):
            if d.isna().any() or len(d) != 95:
                raise SystemExit(f"{dataset} root {root}: unpaired rows in {name}")
            m, lo, hi, x = ci(d)
            res[name] = (m, lo, hi)
            rows.append({"dataset": dataset, "root": root, "H": H, "contrast": name, "mean_pp": 100 * m,
                         "ci95_lo_pp": 100 * lo, "ci95_hi_pp": 100 * hi, "per_seed_pp": np.round(100 * x, 2).tolist()})
            out.append(f"{name}: {100 * m:+.2f} ({100 * lo:+.2f}; {100 * hi:+.2f}), per seed {np.round(100 * x, 2).tolist()}")
        if dataset == BLOOD:
            m, lo, _ = res["tri_all_minus_fltrust_parity"]
            native = res["tri_all_minus_fltrust_native"][0]
            c1, c2 = lo <= 0, m < 0.5 * native
            verdict = "AT RISK" if c1 or c2 else "not at risk"
            out.append(f"Rule: CI95 not entirely above zero {c1}; mean below half of native ({100 * 0.5 * native:+.2f}) {c2}: {verdict}")
        t = pd.DataFrame({"fltrust_native": fl, "fltrust_parity": par, "tri_all": ta}).groupby(level=["alpha", "attack_type"]).mean()
        t["parity_minus_native"] = t.fltrust_parity - t.fltrust_native
        t["tri_all_minus_parity"] = t.tri_all - t.fltrust_parity
        cells.append(t.reset_index().assign(dataset=dataset, root=root, H=H))
        by_alpha = (t.groupby(level="alpha")[["parity_minus_native", "tri_all_minus_parity"]].mean() * 100).round(2)
        n = int((t.tri_all_minus_parity > 0).sum())
        out += ["By alpha (p.p.):", by_alpha.to_string(), f"Cells with tri_all above FLTrust parity: {n} of 19", ""]
    pd.DataFrame(rows).to_csv(f"{OUT}/analysis/contrasts.csv", index=False)
    pd.concat(cells).to_csv(f"{OUT}/analysis/cells.csv", index=False)
    text = "\n".join(out)
    with open(f"{OUT}/analysis/analysis.txt", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
