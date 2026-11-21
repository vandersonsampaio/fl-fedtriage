"""
Summarizes the trigger backdoor grid (`--attacks backdoor_trigger` in `scripts/triage_grid.py`): for
each dataset, system and α, the mean clean accuracy and attack success rate (ASR) at round H over
the seeds, and the number of seeds with ASR below `--blocked`.

Usage:
  venv/bin/python -m scripts.backdoor_summary --raw results/backdoor/raw --out results/backdoor
"""

import argparse
import glob
import os

import pandas as pd


def summarize(raw: str, H: int, blocked: float) -> pd.DataFrame:
    files = glob.glob(f"{raw}/*/*/a*_backdoor_trigger_seed*.csv")
    d = pd.concat(pd.read_csv(f) for f in files)
    d = d[d["round"] == H]
    g = d.groupby(["dataset", "system", "alpha"])
    return pd.DataFrame({
        "n_seeds": g.size(),
        "accuracy": g.accuracy.mean(),
        "asr": g.asr.mean(),
        "asr_min": g.asr.min(),
        "asr_max": g.asr.max(),
        "seeds_blocked": g.asr.apply(lambda s: int((s < blocked).sum())),
    }).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--H", type=int, default=150)
    p.add_argument("--blocked", type=float, default=0.1, help="ASR below which a seed counts as blocked")
    a = p.parse_args()
    s = summarize(a.raw, a.H, a.blocked)
    os.makedirs(a.out, exist_ok=True)
    s.to_csv(f"{a.out}/backdoor_summary.csv", index=False)
    pd.set_option("display.width", 200)
    lines = []
    for ds, part in s.groupby("dataset"):
        w = part.pivot(index="system", columns="alpha", values=["accuracy", "asr"]).round(3)
        lines += [f"=== {ds}: accuracy and ASR at H = {a.H} ===", w.to_string(), ""]
    text = "\n".join(lines)
    with open(f"{a.out}/backdoor_summary.txt", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
