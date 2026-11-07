"""
Per-attack separability of each TRIAGE signal, from the per-client logs written by
`scripts/triage_grid.py --log`. Every signal is oriented so that lower is more suspicious, so the
AUC is computed with -value as the score for is_byz (0.5 = no separation, 1 = perfect).

For each (dataset, α, attack, signal): the AUC pooled over rounds ≥ 2 and seeds, and the mean of
the per-round AUCs. Also the Spearman correlation between signals, pooled over all rows.

Usage:
  venv/bin/python -m scripts.signal_auc --raw results/<experiment>/raw --out results/<experiment>
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.triage.signals import SIGNALS


def _auc(y, x):
    ok = 0 < y.sum() < len(y) and np.isfinite(x).all() and np.std(x) > 0
    return roc_auc_score(y, -x) if ok else np.nan


def load_logs(raw: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(f"{raw}/*/*/*.log.csv.gz")):
        d = pd.read_csv(f)
        dataset, system = f.split(os.sep)[-3:-1]
        name = os.path.basename(f)[1:-len(".log.csv.gz")]
        alpha, rest = name.split("_", 1)
        attack, seed = rest.rsplit("_seed", 1)
        frames.append(d.assign(dataset=dataset, system=system, alpha=float(alpha), attack_type=attack,
                               seed=int(seed)))
    return pd.concat(frames, ignore_index=True)


def analyze(raw: str, out: str) -> None:
    d = load_logs(raw)
    d = d[d["round"] >= 2]
    signals = [s for s in SIGNALS if s in d.columns]
    rows = []
    for (dataset, alpha, attack), g in d.groupby(["dataset", "alpha", "attack_type"]):
        y = g["is_byz"].astype(int).values
        for s in signals:
            per_round = [_auc(r["is_byz"].astype(int).values, r[s].values) for _, r in g.groupby(["seed", "round"])]
            rows.append({"dataset": dataset, "alpha": alpha, "attack_type": attack, "signal": s,
                         "auc_pooled": _auc(y, g[s].values), "auc_round_mean": np.nanmean(per_round)})
    auc = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    auc.to_csv(f"{out}/signal_auc.csv", index=False)

    lines = []
    pd.set_option("display.width", 220)
    for dataset, g in auc.groupby("dataset"):
        piv = g.pivot_table(index=["attack_type", "alpha"], columns="signal", values="auc_round_mean")[signals]
        piv["best"] = piv.idxmax(axis=1)
        lines.append(f"=== {dataset}: mean per-round AUC (rounds >= 2) ===\n{piv.round(3).to_string()}")
        by_attack = g.groupby(["attack_type", "signal"])["auc_round_mean"].mean().unstack()[signals]
        lines.append(f"\n--- {dataset}: mean over α ---\n{by_attack.round(3).to_string()}")
        corr = d[d["dataset"] == dataset][signals].corr(method="spearman")
        lines.append(f"\n--- {dataset}: Spearman between signals (all rows) ---\n{corr.round(2).to_string()}\n")
    text = "\n".join(lines)
    with open(f"{out}/signal_auc.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    analyze(a.raw, a.out)


if __name__ == "__main__":
    main()
