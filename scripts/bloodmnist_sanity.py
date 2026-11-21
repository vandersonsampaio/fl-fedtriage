"""
BloodMNIST sanity check in the in-house framework: FedAvg with 10 clients and no attack, for the
logistic model and the CNN, on every split of `data/prepare_bloodmnist.py`, plus a centralized
reference (one client holding the union of the 10 clients' training data). Records the global
accuracy of every round.

Usage (CPU):
  CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.bloodmnist_sanity --model logistic
  CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.bloodmnist_sanity --model cnn
  CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.bloodmnist_sanity --model cnn --rounds 150 --splits iid \
      --out results/bloodmnist_ceiling
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.fl.federated_learner import FederatedLearner, ParticipantData
from src.utils.data_loader import load_dataset_participants

OUT = "results/bloodmnist_sanity"
INPUT_SHAPE = (28, 28, 3)
N_CLASSES = 8
SPLITS = ["iid", "non_iid", "non_iid_a0.1", "non_iid_a0.05"]


def pooled(participants):
    return [ParticipantData(
        id="pooled",
        X_train=np.concatenate([p.X_train for p in participants]),
        y_train=np.concatenate([p.y_train for p in participants]),
        X_test=np.concatenate([p.X_test for p in participants]),
        y_test=np.concatenate([p.y_test for p in participants]),
    )]


def run(model: str, n_rounds: int, seed: int, out: str = OUT, splits=SPLITS) -> None:
    os.makedirs(out, exist_ok=True)
    rows = []
    for split in splits:
        participants, _ = load_dataset_participants("bloodmnist", split, 10)
        setups = [("fedavg10", participants)] + ([("centralized", pooled(participants))] if split == "iid" else [])
        for setup, clients in setups:
            t0 = time.time()
            learner = FederatedLearner(n_rounds=n_rounds, aggregation="fedavg", n_classes=N_CLASSES,
                                       model_type=model, input_shape=INPUT_SHAPE, seed=seed)
            for r in learner.train(clients, verbose=False):
                rows.append({"model": model, "split": split, "setup": setup, "seed": seed,
                             "round": r.round_num, "accuracy": r.global_accuracy})
            print(f"model={model} split={split} setup={setup} acc={rows[-1]['accuracy']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    suffix = "" if list(splits) == SPLITS else "_" + "-".join(splits)
    pd.DataFrame(rows).to_csv(f"{out}/{model}_seed{seed}{suffix}.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["logistic", "cnn"])
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=OUT)
    p.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    a = p.parse_args()
    run(a.model, a.rounds, a.seed, a.out, a.splits)


if __name__ == "__main__":
    main()
