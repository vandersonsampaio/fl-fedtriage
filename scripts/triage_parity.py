"""
Runs the AdaAggRL-skeleton variants of B2.6 through `TriageLearner` on the B2.6 grid (MNIST,
α ∈ {0.5, 0.1, 0.05}, 7 attacks, clients 0 and 1 Byzantine, 15 rounds) and writes CSVs in the
format of `scripts/b26_decomposicao.py`, to be compared with fl-gradf by
`scripts/parity_fl_gradf.py`.

Usage (CPU):
  CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.triage_parity --variant sr_bin --seed 52
  venv/bin/python -m scripts.parity_fl_gradf --gradf ../fl-fedmed --new results/triage_parity/raw sr_bin_seed52.csv
"""

import argparse
import os
import time

import pandas as pd

from src.triage import TriageLearner, TriageRule
from src.triage.signals import INVERSION, SERVER_COSINE

OUT = "results/triage_parity/raw"
INPUT_SHAPE = (28, 28, 1)
ALPHAS = [0.5, 0.1, 0.05]
BYZ = [0, 1]
N_ROUNDS = 15

VARIANTS = {
    "sr_only": dict(signals=[INVERSION], b=0.5, weighting="soft", lam=2.0),
    "sr_bin": dict(signals=[INVERSION], b=0.5, weighting="uniform", lam=2.0),
    "sr_b025": dict(signals=[INVERSION], b=0.25, weighting="soft", lam=2.0),
    "cosserver_only": dict(signals=[SERVER_COSINE], b=0.5, weighting="soft", lam=2.0),
}


def run(variant: str, seed: int, out: str = OUT) -> None:
    from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS, _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(out, exist_ok=True)
    path = f"{out}/{variant}_seed{seed}.csv"
    if os.path.exists(path):
        print(f"{path} already exists, skipping")
        return
    rows = []
    for alpha in ALPHAS:
        participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
        for attack in INFORMED_ATTACKS + BLIND_ATTACKS:
            t0 = time.time()
            learner = TriageLearner(n_rounds=N_ROUNDS, n_classes=10, attack_type=attack, byzantine_ids=BYZ,
                                    seed=seed, input_shape=INPUT_SHAPE, rule=TriageRule(**VARIANTS[variant]),
                                    record=False)
            acc = learner.train(participants, root_data=root, verbose=False)[-1].global_accuracy
            rows.append({"alpha": alpha, "attack_type": attack, "variant": variant, "seed": seed, "accuracy": acc})
            print(f"variant={variant} seed={seed} alpha={alpha} attack={attack} acc={acc:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    run(a.variant, a.seed, a.out)


if __name__ == "__main__":
    main()
