"""
Partition BloodMNIST for federated learning, in the same layout as `download_datasets.py`.

Reads the official MedMNIST v2 file (28×28×3, 8 classes; Zenodo record 10519652), expected at
data/raw/medmnist/bloodmnist.npz, and keeps its official train and test splits (the official
validation split is not used). Pixels are scaled to [0, 1].

Writes data/processed/bloodmnist/{iid, non_iid, non_iid_a0.1, non_iid_a0.05}/, where `non_iid`
is Dirichlet α = 0.5, as for MNIST and CIFAR-10.

Usage:
  python data/prepare_bloodmnist.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_datasets import (  # noqa: E402
    _dirichlet_partition, _iid_partition, _non_iid_coefficient, _save_split,
)

NAME = "bloodmnist"
INPUT_SHAPE = (28, 28, 3)
N_CLASSES = 8


def load_npz(path: str):
    d = np.load(path)
    X_train = d["train_images"].astype(np.float32) / 255.0
    X_test = d["test_images"].astype(np.float32) / 255.0
    return X_train, d["train_labels"].ravel().astype(np.int64), X_test, d["test_labels"].ravel().astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description="Partition BloodMNIST for FL")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--extra_alphas", type=float, nargs="*", default=[0.1, 0.05])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(__file__))
    proc_root = os.path.join(data_dir, "processed")
    X_train, y_train, X_test, y_test = load_npz(os.path.join(data_dir, "raw", "medmnist", f"{NAME}.npz"))
    print(f"{NAME}: train={X_train.shape} test={X_test.shape} classes={np.bincount(y_train).tolist()}")

    rng = np.random.default_rng(args.seed)
    splits = [("iid", None), ("non_iid", args.alpha)] + [(f"non_iid_a{a}", a) for a in args.extra_alphas]
    for split_name, alpha in splits:
        print(f"[{split_name}] n_clients={args.n_clients}" + ("" if alpha is None else f" α={alpha}"))
        if alpha is None:
            idx = _iid_partition(X_train, y_train, args.n_clients, rng)
        else:
            idx = _dirichlet_partition(y_train, args.n_clients, alpha, rng)
        _save_split(split_name, NAME, idx, X_train, y_train, X_test, y_test, proc_root, rng=rng)
        print(f"    non-IID coefficient: {_non_iid_coefficient(idx, y_train):.4f}")


if __name__ == "__main__":
    main()
