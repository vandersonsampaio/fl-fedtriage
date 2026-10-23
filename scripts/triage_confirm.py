"""
Runs the tests of `results/f3_confirmation/PREREGISTRATION.md` on the confirmation seeds (62–71).

`scripts/triage_compare.py` reports a two-sided Wilcoxon p and a TOST p. The preregistration asks
for a one-sided Wilcoxon (candidate > reference) with Holm over each family of hypotheses, so this
script computes them from the same per-seed means. For H4 it reports both one-sided tests of the
TOST: the lower one (non-inferiority, difference > −margin) and the upper one (difference < +margin).

It also writes the preregistered descriptive results:
  - root cost: each root-dependent system with `@r20` minus the same system with the canonical root;
  - worst case: per cell, `tri_all` minus the best fixed rule of that cell;
  - the no-attack ceiling (`ceiling_fedavg10`) per α.

Usage:
  venv/bin/python -m scripts.triage_confirm --raw results/f3_confirmation/raw --out results/f3_confirmation
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from scripts.triage_compare import load

SEEDS = list(range(62, 72))
SINGLE = ["cosserver_only", "cos_bin", "sr_bin"]
SINGLE_R20 = ["cosserver_only@r20", "cos_bin@r20", "sr_bin"]
FIXED = ["fltrust", "fedavg", "trimmed_mean", "median", "clustering", "krum"]
ROOT_DEPENDENT = ["tri_all", "cosserver_only", "cos_bin", "fltrust"]
HORIZONS = {"bloodmnist": [150, 250, 300], "mnist": [150]}
PRIMARY_H = {"bloodmnist": 250, "mnist": 150}


def wide(d: pd.DataFrame, H: int) -> pd.DataFrame:
    w = d[(d.H == H) & (d.attack_type != "none")]
    return w.pivot_table(index=["alpha", "attack_type", "seed"], columns="system", values="accuracy")


def per_seed(w: pd.DataFrame, cand: str, ref: str) -> pd.Series:
    m = w[[cand, ref]].dropna()
    return (m[cand] - m[ref]).groupby("seed").mean()


def best_of(w: pd.DataFrame, systems) -> str:
    return w[systems].dropna().mean().idxmax()


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    k = len(order)
    adj = np.maximum.accumulate([(k - i) * v for i, v in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1.0), index=order.index).reindex(p.index)


def family(w: pd.DataFrame, cand: str, refs: dict, label: str) -> pd.DataFrame:
    rows = []
    for hyp, ref in refs.items():
        s = per_seed(w, cand, ref)
        rows.append({"family": label, "hypothesis": hyp, "candidate": cand, "reference": ref,
                     "n_seeds": len(s), "diff_pp": 100 * s.mean(), "seeds_positive": int((s > 0).sum()),
                     "p_one_sided": float(stats.wilcoxon(s, alternative="greater").pvalue)})
    f = pd.DataFrame(rows)
    f["p_holm"] = holm(f.set_index("hypothesis")["p_one_sided"]).values
    f["significant"] = f["p_holm"] < 0.05
    return f


def noninferiority(w: pd.DataFrame, cand: str, ref: str, margin: float) -> dict:
    s = per_seed(w, cand, ref)
    lo = stats.ttest_1samp(s, -margin, alternative="greater").pvalue
    hi = stats.ttest_1samp(s, margin, alternative="less").pvalue
    se = s.std(ddof=1) / np.sqrt(len(s))
    t = stats.t.ppf(0.95, len(s) - 1)
    return {"hypothesis": "H4", "candidate": cand, "reference": ref, "n_seeds": len(s),
            "diff_pp": 100 * s.mean(), "ci90_low_pp": 100 * (s.mean() - t * se),
            "ci90_high_pp": 100 * (s.mean() + t * se), "p_lower": float(lo), "p_upper": float(hi),
            "p_tost": float(max(lo, hi)), "noninferior": bool(lo < 0.05), "equivalent": bool(max(lo, hi) < 0.05)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="results/f3_confirmation/raw")
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--out", default="results/f3_confirmation")
    a = p.parse_args()

    data = {ds: load(ds, [a.raw], HORIZONS[ds]).query("seed in @SEEDS") for ds in HORIZONS}
    lines = []
    pd.set_option("display.width", 250)

    w = wide(data["bloodmnist"], PRIMARY_H["bloodmnist"])
    best, best_r20 = best_of(w, SINGLE), best_of(w, SINGLE_R20)
    primary = family(w, "tri_all", {"H1": best, "H2": "fltrust", "H3": "flame"}, "primary")
    secondary = family(w, "tri_all@r20", {"H1'": best_r20, "H2'": "fltrust@r20", "H3'": "flame"}, "secondary @r20")
    tests = pd.concat([primary, secondary], ignore_index=True)

    wm = wide(data["mnist"], PRIMARY_H["mnist"])
    h4 = pd.DataFrame([noninferiority(wm, "tri_all", best_of(wm, SINGLE), a.margin)])

    root_rows, worst_rows = [], []
    for ds, hs in HORIZONS.items():
        for H in hs:
            w = wide(data[ds], H)
            for s in ROOT_DEPENDENT:
                diff = per_seed(w, f"{s}@r20", s)
                root_rows.append({"dataset": ds, "H": H, "system": s, "diff_pp": 100 * diff.mean(),
                                  "seeds_negative": int((diff < 0).sum())})
            cell = w.groupby(["alpha", "attack_type"]).mean()
            best_rule = cell[FIXED].idxmax(axis=1)
            for (alpha, attack), rule in best_rule.items():
                worst_rows.append({"dataset": ds, "H": H, "alpha": alpha, "attack_type": attack,
                                   "best_fixed_rule": rule,
                                   "diff_pp": 100 * (cell.loc[(alpha, attack), "tri_all"] - cell.loc[(alpha, attack), rule])})
    root, worst = pd.DataFrame(root_rows), pd.DataFrame(worst_rows)

    ceil_rows = []
    for ds, d in data.items():
        H = PRIMARY_H[ds]
        c = d[(d.H == H) & (d.system == "ceiling_fedavg10")].groupby("alpha")["accuracy"].mean()
        t = d[(d.H == H) & (d.system == "tri_all")].groupby("alpha")["accuracy"].mean()
        for alpha in c.index:
            ceil_rows.append({"dataset": ds, "H": H, "alpha": alpha, "ceiling_fedavg10": c[alpha],
                              "tri_all": t[alpha], "gap_pp": 100 * (t[alpha] - c[alpha])})
    ceiling = pd.DataFrame(ceil_rows)

    os.makedirs(a.out, exist_ok=True)
    tests.to_csv(f"{a.out}/confirm_tests.csv", index=False)
    h4.to_csv(f"{a.out}/confirm_h4.csv", index=False)
    root.to_csv(f"{a.out}/confirm_root_cost.csv", index=False)
    worst.to_csv(f"{a.out}/confirm_worst_case.csv", index=False)
    ceiling.to_csv(f"{a.out}/confirm_ceiling.csv", index=False)

    lines.append("=== H1–H3 and H1'–H3' (BloodMNIST, H = 250; one-sided Wilcoxon, Holm per family) ===")
    lines.append(tests.round(5).to_string(index=False))
    lines.append(f"\n=== H4 (MNIST, H = 150; margin ±{100 * a.margin:g} p.p.) ===")
    lines.append(h4.round(5).to_string(index=False))
    lines.append("\n=== root cost: system@r20 − system (p.p.) ===")
    lines.append(root.pivot_table(index="system", columns=["dataset", "H"], values="diff_pp").round(2).to_string())
    lines.append("\n=== worst case: tri_all − best fixed rule of each cell, five worst per dataset and H (p.p.) ===")
    lines.append(worst.sort_values("diff_pp").groupby(["dataset", "H"]).head(5).round(2).to_string(index=False))
    lines.append("\n=== no-attack ceiling per α ===")
    lines.append(ceiling.round(4).to_string(index=False))
    text = "\n".join(lines)
    with open(f"{a.out}/confirm.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
