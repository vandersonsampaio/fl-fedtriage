"""
Grid runner for TRIAGE and its references: one job = one system on one cell (dataset, α, attack)
and seed. Writes the global accuracy of every round to
<out>/raw/<dataset>/<system>/a<alpha>_<attack>_seed<seed>.csv and, with --log, the per-client
log (signals, decision, is_byz) next to it as .log.csv.gz. Existing outputs are skipped.

Systems:
  rules       fedavg, fltrust, trimmed_mean, clustering, median, krum (server update always
              computed, as in fl-gradf's C0b)
  ceilings    ceiling_fedavg10 (no attack), oracle_fedavg8 (FedAvg over the honest clients only,
              evaluated on all 10), both once per α with attack "none"; oracle_uniform (attacked
              run in which exactly the honest clients are averaged, with uniform weights)
  TRIAGE      every entry of CONFIGS (a TriageRule configuration)

Cells: α ∈ {0.5, 0.1, 0.05} × 7 attacks, without fltrust_aligned at α ∈ {0.1, 0.05} (19 cells);
clients 0 and 1 Byzantine; root of 100 samples drawn with the seed from the server validation set.

Root variants (`--root_size`, `--root_client`, `--root_noise`): the root can be smaller, drawn from
the training data of one honest client (an anchor client, which keeps its data) instead of the
server validation set, or have a fraction of its labels replaced by a different random class. A
non-default root appends a tag to the system name, e.g. `cos_bin@r20` or `cos_bin@c2-r100-n0.2`.

Usage (CPU):
  venv/bin/python -m scripts.triage_grid cell --dataset mnist --system sr_bin --alpha 0.1 \\
      --attack sign_flipping --seed 42 --rounds 150 --out results/<experiment>
  venv/bin/python -m scripts.triage_grid jobs --dataset mnist --systems fltrust sr_bin \\
      --seeds 42 43 --rounds 150 --out results/<experiment> > jobs.txt
  bash scripts/launch_grid.sh <experiment> jobs.txt -P 12
"""

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.triage.signals import (
    INVERSION, MEDIAN_COSINE, NEG_MEDIAN_DISTANCE, NEG_NORM, SERVER_COSINE, TEMPORAL_COSINE,
)

DATASETS = {
    "mnist": dict(n_classes=10, input_shape=(28, 28, 1)),
    "bloodmnist": dict(n_classes=8, input_shape=(28, 28, 3)),
}
ALPHAS = [0.5, 0.1, 0.05]
ATTACKS = ["fltrust_aligned", "trim_attack", "krum_collusion", "low_mag_backdoor",
           "sign_flipping", "gaussian_noise", "label_flipping"]
EXCLUDED = {(0.1, "fltrust_aligned"), (0.05, "fltrust_aligned")}
BYZ = [0, 1]
ROOT_SIZE = 100

RULES = ["fedavg", "fltrust", "trimmed_mean", "clustering", "median", "krum"]
CEILINGS = ["ceiling_fedavg10", "oracle_fedavg8", "oracle_uniform"]
NO_ATTACK = ("ceiling_fedavg10", "oracle_fedavg8")
CHEAP = [MEDIAN_COSINE, NEG_NORM, NEG_MEDIAN_DISTANCE, TEMPORAL_COSINE]
ALL_SIGNALS = [INVERSION, SERVER_COSINE] + CHEAP

CONFIGS: Dict[str, dict] = {
    "sr_only": dict(signals=[INVERSION], b=0.5, weighting="soft", lam=2.0),
    "sr_bin": dict(signals=[INVERSION], b=0.5, weighting="uniform", lam=2.0),
    "sr_b025": dict(signals=[INVERSION], b=0.25, weighting="soft", lam=2.0),
    "cosserver_only": dict(signals=[SERVER_COSINE], b=0.5, weighting="soft", lam=2.0),
    "cos_bin": dict(signals=[SERVER_COSINE], b=0.5, weighting="uniform", lam=2.0),
    "tri_any": dict(signals=[INVERSION, SERVER_COSINE], combine="any", b=0.5, weighting="uniform", lam=2.0),
    "tri_all": dict(signals=[INVERSION, SERVER_COSINE], combine="all", b=0.5, weighting="uniform", lam=2.0),
    "tri_conf": dict(signals=[INVERSION, SERVER_COSINE], combine="confidence", b=0.5, weighting="uniform",
                     lam=2.0),
    "tri_rank": dict(signals=[INVERSION, SERVER_COSINE], combine="min_rank", b=0.5, weighting="uniform",
                     lam=2.0),
    "cos_temp_any": dict(signals=[SERVER_COSINE, TEMPORAL_COSINE], combine="any", b=0.5, weighting="uniform",
                         lam=2.0),
    "cheap_any": dict(signals=[NEG_MEDIAN_DISTANCE, TEMPORAL_COSINE], combine="any", b=0.5, weighting="uniform",
                      lam=2.0),
    "cos_mad2": dict(signals=[SERVER_COSINE], threshold="mad", k=2.0, weighting="uniform", lam=2.0),
    "cos_mad3": dict(signals=[SERVER_COSINE], threshold="mad", k=3.0, weighting="uniform", lam=2.0),
    "tri_all_mad2": dict(signals=[INVERSION, SERVER_COSINE], combine="all", threshold="mad", k=2.0,
                         weighting="uniform", lam=2.0),
    "tri_conf_mad2": dict(signals=[INVERSION, SERVER_COSINE], combine="confidence", threshold="mad", k=2.0,
                          weighting="uniform", lam=2.0),
}
SYSTEMS = RULES + CEILINGS + list(CONFIGS)


def cells():
    return [(a, t) for a in ALPHAS for t in ATTACKS if (a, t) not in EXCLUDED]


def split_name(alpha: float) -> str:
    return "non_iid" if alpha == 0.5 else f"non_iid_a{alpha}"


def root_tag(size: int = ROOT_SIZE, client: Optional[int] = None, noise: float = 0.0) -> str:
    if size == ROOT_SIZE and client is None and noise == 0.0:
        return ""
    parts = ([f"c{client}"] if client is not None else []) + [f"r{size}"] + ([f"n{noise}"] if noise else [])
    return "@" + "-".join(parts)


def make_root(participants, server_root: Dict, size: int, client: Optional[int], noise: float, n_classes: int,
              seed: int) -> Dict:
    """Root dataset: `size` samples from the server validation set or, with `client`, from that
    client's training data; then a fraction `noise` of the labels replaced by another class."""
    rng = np.random.default_rng(seed)
    if client is None:
        X, y = server_root["X"], server_root["y"]
    else:
        if client in BYZ:
            raise ValueError(f"the anchor client must be honest, got {client}")
        X, y = participants[client].X_train, participants[client].y_train
    idx = rng.choice(len(y), size=min(size, len(y)), replace=False)
    X, y = X[idx], y[idx].copy()
    if noise:
        flip = rng.random(len(y)) < noise
        y[flip] = (y[flip] + rng.integers(1, n_classes, flip.sum())) % n_classes
    return {"X": X, "y": y}


def output_path(out: str, dataset: str, system: str, alpha: float, attack: str, seed: int) -> str:
    return f"{out}/raw/{dataset}/{system}/a{alpha}_{attack}_seed{seed}.csv"


def build_rule_learner():
    from src.experiments.exp10_selector_comparison import _make_arsenal_strategy
    from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    class RuleLearner(AttackedFederatedLearner):
        def __init__(self, *args, rule: str, **kwargs):
            kwargs.setdefault("aggregation", "fedavg")
            super().__init__(*args, **kwargs)
            self.rule = rule

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            updates, is_byz = compute_param_updates_auto(self, participants, active, root_data)
            root = root_data or self._carve_root(participants[0])
            server_update = self._make_model(participants[0].n_features).fit(root["X"], root["y"])
            delta, _ = _make_arsenal_strategy(self.rule, sum(is_byz)).aggregate(
                updates, sample_sizes=[p.n_train for p in participants], server_update=server_update)
            self._global_params = self._global_params + delta
            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc)

    return RuleLearner


def build_oracle_learner():
    from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    class OracleUniformLearner(AttackedFederatedLearner):
        """Attacked run in which exactly the honest clients are averaged, with uniform weights."""

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("aggregation", "fedavg")
            super().__init__(*args, **kwargs)

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            updates, _ = compute_param_updates_auto(self, participants, active, root_data)
            honest = [u for i, u in enumerate(updates) if i not in self.byzantine_ids]
            self._global_params = self._global_params + np.mean(honest, axis=0)
            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc)

    class OracleFedAvg8Learner(AttackedFederatedLearner):
        """FedAvg over the honest clients only, without attack; accuracy measured on `eval_on`."""

        def __init__(self, *args, eval_on, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_on = eval_on

        def _run_round(self, round_num, participants, root_data):
            rr = super()._run_round(round_num, participants, root_data)
            m = self._make_model(participants[0].n_features)
            rr.global_accuracy = float(np.mean([m.accuracy(p.X_test, p.y_test) for p in self.eval_on]))
            return rr

    return OracleUniformLearner, OracleFedAvg8Learner


def run_job(dataset: str, system: str, alpha: float, attack: str, seed: int, rounds: int, out: str,
            model: str = "logistic", log: bool = False, root_size: int = ROOT_SIZE,
            root_client: Optional[int] = None, root_noise: float = 0.0) -> Optional[str]:
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.triage import TriageLearner, TriageRule
    from src.utils.data_loader import load_dataset_participants

    name = system + root_tag(root_size, root_client, root_noise)
    path = output_path(out, dataset, name, alpha, attack, seed)
    if os.path.exists(path):
        print(f"{path} already exists, skipping")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    participants, server_root = load_dataset_participants(dataset, split_name(alpha), 10)
    n_classes = DATASETS[dataset]["n_classes"]
    root = make_root(participants, server_root, root_size, root_client, root_noise, n_classes, seed)
    common = dict(n_rounds=rounds, n_classes=n_classes, n_classes_data=n_classes, seed=seed,
                  model_type=model, input_shape=DATASETS[dataset]["input_shape"])
    t0 = time.time()
    learner = None
    if system in RULES:
        learner = build_rule_learner()(attack_type=attack, byzantine_ids=BYZ, rule=system, **common)
        history = learner.train(participants, root_data=root, verbose=False)
    elif system == "ceiling_fedavg10":
        learner = AttackedFederatedLearner(attack_type="none", byzantine_ids=BYZ, aggregation="fedavg", **common)
        history = learner.train(participants, root_data=root, verbose=False)
    elif system == "oracle_fedavg8":
        honest = [p for i, p in enumerate(participants) if i not in BYZ]
        learner = build_oracle_learner()[1](attack_type="none", byzantine_ids=[], aggregation="fedavg",
                                            eval_on=participants, **common)
        history = learner.train(honest, root_data=root, verbose=False)
    elif system == "oracle_uniform":
        learner = build_oracle_learner()[0](attack_type=attack, byzantine_ids=BYZ, **common)
        history = learner.train(participants, root_data=root, verbose=False)
    elif system in CONFIGS:
        observe = [s for s in ALL_SIGNALS if s != INVERSION or model == "logistic"] if log else []
        learner = TriageLearner(attack_type=attack, byzantine_ids=BYZ, rule=TriageRule(**CONFIGS[system]),
                                record=log, observe=observe, **common)
        history = learner.train(participants, root_data=root, verbose=False)
    else:
        raise ValueError(f"unknown system {system!r}")
    accs = [h.global_accuracy for h in history]
    sec = time.time() - t0
    df = pd.DataFrame({"dataset": dataset, "model": model, "system": name, "alpha": alpha,
                       "attack_type": attack, "seed": seed, "round": np.arange(1, len(accs) + 1),
                       "accuracy": accs, "sec": sec})
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    if log and getattr(learner, "log", None):
        pd.DataFrame(learner.log).to_csv(path.replace(".csv", ".log.csv.gz"), index=False)
    os.replace(tmp, path)
    print(f"{dataset} {system} α={alpha} {attack} seed={seed}: acc={accs[-1]:.4f} ({sec:.0f}s)", flush=True)
    return path


def jobs(datasets: List[str], systems: List[str], seeds: List[int], rounds: int, out: str,
         model: str, log: bool, attacks: Optional[List[str]] = None, root_size: int = ROOT_SIZE,
         root_client: Optional[int] = None, root_noise: float = 0.0) -> List[str]:
    lines = []
    base = 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 venv/bin/python -m scripts.triage_grid cell'
    tag = root_tag(root_size, root_client, root_noise)
    root_args = (f" --root_size {root_size}" + (f" --root_client {root_client}" if root_client is not None else "")
                 + (f" --root_noise {root_noise}" if root_noise else "")) if tag else ""
    for dataset in datasets:
        for system in systems:
            for seed in seeds:
                todo = [(a, "none") for a in ALPHAS] if system in NO_ATTACK else cells()
                for alpha, attack in todo:
                    if attacks and attack not in attacks + ["none"]:
                        continue
                    if os.path.exists(output_path(out, dataset, system + tag, alpha, attack, seed)):
                        continue
                    lines.append(f"{base} --dataset {dataset} --system {system} --alpha {alpha} --attack {attack} "
                                 f"--seed {seed} --rounds {rounds} --out {out} --model {model}"
                                 + root_args + (" --log" if log else ""))
    return lines


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    c.add_argument("--system", required=True, choices=SYSTEMS)
    c.add_argument("--alpha", type=float, required=True)
    c.add_argument("--attack", required=True, choices=ATTACKS + ["none"])
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--rounds", type=int, default=150)
    c.add_argument("--model", default="logistic", choices=["logistic", "cnn"])
    c.add_argument("--log", action="store_true")
    c.add_argument("--out", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--datasets", nargs="+", default=["mnist"], choices=sorted(DATASETS))
    j.add_argument("--systems", nargs="+", required=True, choices=SYSTEMS)
    j.add_argument("--seeds", type=int, nargs="+", required=True)
    j.add_argument("--attacks", nargs="+", default=None, choices=ATTACKS)
    j.add_argument("--rounds", type=int, default=150)
    j.add_argument("--model", default="logistic", choices=["logistic", "cnn"])
    j.add_argument("--log", action="store_true")
    j.add_argument("--out", required=True)
    for q in (c, j):
        q.add_argument("--root_size", type=int, default=ROOT_SIZE)
        q.add_argument("--root_client", type=int, default=None)
        q.add_argument("--root_noise", type=float, default=0.0)
    a = p.parse_args()
    root = dict(root_size=a.root_size, root_client=a.root_client, root_noise=a.root_noise)
    if a.cmd == "cell":
        run_job(a.dataset, a.system, a.alpha, a.attack, a.seed, a.rounds, a.out, a.model, a.log, **root)
    else:
        print("\n".join(jobs(a.datasets, a.systems, a.seeds, a.rounds, a.out, a.model, a.log, a.attacks, **root)))


if __name__ == "__main__":
    main()
