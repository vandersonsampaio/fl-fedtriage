"""
Block E2: how much of a client's data the server can recover from the update it inspects.

The server of TRIAGE runs gradient inversion on every update to compute S_R. This script measures,
on FedAvg without attack (so every client is honest and included), at checkpoint rounds:

  1. sr_recon   what S_R's own reconstruction recovers. For each reconstructed sample, the cosine
                to its nearest neighbor among the client's training samples (own) and among an
                equal-size sample of the other clients' training samples (other). Leakage specific
                to the client = own − other.
  2. prototype  what a curious server gets from the same update with no optimization. In the
                logistic model, ΔW[:, k] / Δb[k] is a weighted mean of the client's samples tied to
                class k. Cosine of this prototype to the client's class-k mean (own) and to the
                other clients' pooled class-k mean (other), over classes the client has (≥ 5
                samples) and with |Δb[k]| above 1% of max |Δb|.
  3. labels     Spearman correlation between Δb[k] (the update is −lr × gradient, so present classes
                push Δb[k] up) and the client's class proportions (label
                distribution inference), and whether the classes the client lacks are exactly
                those with Δb[k] < 0.

Samples are compared after centering each pixel vector on the server validation mean, so that
"looking like an average image" does not count as leakage.

Usage (CPU):
  venv/bin/python -m scripts.inversion_leakage --dataset bloodmnist --alpha 0.1 --seed 120 \\
      --rounds 150 --checkpoints 1 10 50 150 --out results/inversion_leakage
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from scripts.triage_grid import BYZ, DATASETS, make_root, split_name

MIN_CLASS = 5
POOL = 2000


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def nn_cos(queries: np.ndarray, pool: np.ndarray) -> np.ndarray:
    return (unit(queries) @ unit(pool).T).max(axis=1)


def measure(round_num, params, update, p, others_X, others_y, center, n_features, n_classes, local_lr, seed):
    from src.defense.adaaggrl_agent import reconstruct_client_distribution

    K = n_classes
    rows: List[Dict] = []
    theta = params + update
    W = theta[: n_features * K].reshape(n_features, K).astype(np.float32)
    b = theta[n_features * K:].astype(np.float32)
    rec, s_r = reconstruct_client_distribution(W, b, update, local_lr, n_features, K, seed=round_num)
    rng = np.random.default_rng(seed * 1000 + round_num)
    own = p.X_train[rng.choice(len(p.X_train), size=min(POOL, len(p.X_train)), replace=False)] - center
    oth = others_X[rng.choice(len(others_X), size=len(own), replace=False)] - center
    rc = rec - center
    o, t = nn_cos(rc, own), nn_cos(rc, oth)
    base = {"round": round_num, "client": p.id, "s_r": s_r}
    rows.append({**base, "measure": "sr_recon", "own": o.mean(), "other": t.mean(), "gap": (o - t).mean()})

    dW = update[: n_features * K].reshape(n_features, K)
    db = update[n_features * K:]
    counts = np.bincount(p.y_train, minlength=K)
    own_c, oth_c = [], []
    for k in range(K):
        if counts[k] < MIN_CLASS or abs(db[k]) < 0.01 * np.abs(db).max():
            continue
        proto = dW[:, k] / db[k] - center
        own_mean = p.X_train[p.y_train == k].mean(axis=0) - center
        sel = others_y == k
        if sel.sum() < MIN_CLASS:
            continue
        oth_mean = others_X[sel].mean(axis=0) - center
        own_c.append(float(unit(proto) @ unit(own_mean)))
        oth_c.append(float(unit(proto) @ unit(oth_mean)))
    if own_c:
        rows.append({**base, "measure": "prototype", "own": np.mean(own_c), "other": np.mean(oth_c),
                     "gap": np.mean(np.array(own_c) - np.array(oth_c)), "n_classes_used": len(own_c)})
    prop = counts / counts.sum()
    rho = stats.spearmanr(db, prop).statistic if prop.std() > 0 else np.nan
    absent = counts == 0
    exact = bool(((db < 0) == absent).all()) if absent.any() else np.nan
    rows.append({**base, "measure": "labels", "own": rho, "other": np.nan, "gap": np.nan,
                 "absent_classes": int(absent.sum()), "absent_exact": exact})
    return rows


def run(dataset, alpha, seed, rounds, checkpoints, out):
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    path = f"{out}/raw/{dataset}/a{alpha}_seed{seed}.csv"
    if os.path.exists(path):
        print(f"{path} exists, skipping")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    participants, server_root = load_dataset_participants(dataset, split_name(alpha), 10)
    n_classes = DATASETS[dataset]["n_classes"]
    root = make_root(participants, server_root, 100, None, 0.0, n_classes, seed)
    center = server_root["X"].mean(axis=0)
    rows: List[Dict] = []

    class Probe(AttackedFederatedLearner):
        def _run_round(self, round_num, parts, root_data):
            if round_num in checkpoints:
                state = np.random.get_state()
                updates, _ = self._compute_param_updates(parts, False)
                np.random.set_state(state)
                for i, p in enumerate(parts):
                    oX = np.concatenate([q.X_train for j, q in enumerate(parts) if j != i])
                    oy = np.concatenate([q.y_train for j, q in enumerate(parts) if j != i])
                    rows.extend(measure(round_num, self._global_params, updates[i], p, oX, oy, center,
                                        p.n_features, n_classes, self.local_lr, seed))
            return super()._run_round(round_num, parts, root_data)

    learner = Probe(n_rounds=rounds, n_classes=n_classes, n_classes_data=n_classes, seed=seed, model_type="logistic",
                    input_shape=DATASETS[dataset]["input_shape"], attack_type="none", byzantine_ids=BYZ,
                    aggregation="fedavg")
    learner.train(participants, root_data=root, verbose=False)
    df = pd.DataFrame(rows).assign(dataset=dataset, alpha=alpha, seed=seed)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(df.groupby(["round", "measure"])[["own", "other", "gap"]].mean().round(3).to_string())


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    a.add_argument("--alpha", type=float, required=True)
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--rounds", type=int, default=150)
    a.add_argument("--checkpoints", type=int, nargs="+", default=[1, 10, 50, 150])
    a.add_argument("--out", required=True)
    x = a.parse_args()
    run(x.dataset, x.alpha, x.seed, x.rounds, set(x.checkpoints), x.out)


if __name__ == "__main__":
    main()
