"""
Determinism check: |Δ| = 0 for the same seed.

Runs every baseline system on one cell (dataset, α, attack, seed) and records the global
accuracy of every round. The same seed must give exactly the same trajectory:
  - twice in the same process (state left over by one run must not leak into the next);
  - in a fresh process, with the systems in reverse order (no dependence on execution order).

Usage (CPU):
  CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.determinism_check run --order fwd --repeat 2 --out results/determinism/fwd.json
  CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.determinism_check run --order rev --out results/determinism/rev.json
  venv/bin/python -m scripts.determinism_check check results/determinism/fwd.json results/determinism/rev.json
"""

import argparse
import json
import os
import time

SYSTEMS = [
    "fedavg", "fltrust", "median", "trimmed_mean", "krum", "clustering",
    "random", "oracle", "gradf", "fedstrategist", "adaaggrl", "tars", "adabfl",
]
NEEDS_PRETRAIN = {"oracle", "gradf", "fedstrategist"}


def run_systems(systems, alpha, attack, seed, n_rounds, n_clients=10, byzantine_fraction=0.2, n_classes=10, root_size=100):
    from src.defense.rl_selector import RLDefenseSelector
    from src.experiments.exp1_baseline import (
        GroundTruthClassifierStub, GroundTruthOracleLearner, OracleSelector, RandomSelector, train_gradf_models,
    )
    from src.experiments.exp4_adaptive import AdaBFLLearner, TARSLearner
    from src.experiments.exp9_dominance_grid import _split_name
    from src.experiments.exp10_selector_comparison import (
        DATASET_INPUT_SHAPE, AdaAggRLGridLearner, FedStrategistGridLearner,
    )
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.fl.gradf_learner import GRADFFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    participants, root_data = load_dataset_participants(
        "mnist", _split_name(alpha), n_clients, root_size=root_size, root_seed=seed,
    )
    byz = list(range(max(1, int(n_clients * byzantine_fraction))))
    labels = ["none", attack]
    common = dict(n_rounds=n_rounds, n_classes=n_classes, attack_type=attack, byzantine_ids=byz, seed=seed)

    classifier = selector = oracle_map = None
    if NEEDS_PRETRAIN & set(systems):
        classifier, selector, _l, oracle_map = train_gradf_models(
            participants, attack, byzantine_fraction, n_classes, n_rounds=8, seed=seed,
        )

    def build(name):
        if name in ("fedavg", "fltrust", "median", "trimmed_mean", "krum", "clustering"):
            return AttackedFederatedLearner(aggregation=name, **common)
        if name == "random":
            return GRADFFederatedLearner(
                aggregation="fedavg", classifier=GroundTruthClassifierStub(),
                selector=RandomSelector(RLDefenseSelector().actions, seed=seed), attack_labels=labels, **common,
            )
        if name == "oracle":
            return GroundTruthOracleLearner(
                aggregation="fedavg", classifier=GroundTruthClassifierStub(),
                selector=OracleSelector(oracle_map), attack_labels=labels, **common,
            )
        if name == "gradf":
            return GRADFFederatedLearner(
                aggregation="fedavg", classifier=classifier, selector=selector, attack_labels=labels, **common,
            )
        if name == "fedstrategist":
            return FedStrategistGridLearner(variant="b", shared_classifier=classifier, n_labels=len(labels), **common)
        if name == "adaaggrl":
            return AdaAggRLGridLearner(
                input_shape=DATASET_INPUT_SHAPE["mnist"], feature_extractor="random", dataset="mnist", **common,
            )
        if name == "tars":
            return TARSLearner(attack_sequence=[attack], **common)
        if name == "adabfl":
            return AdaBFLLearner(attack_sequence=[attack], **common)
        raise ValueError(name)

    out = {}
    for name in systems:
        t0 = time.time()
        hist = build(name).train(participants, root_data=root_data, verbose=False)
        out[name] = [float(r.global_accuracy) for r in hist]
        print(f"{name:14s} final={out[name][-1]:.6f} ({time.time() - t0:.1f}s)", flush=True)
    return out


def cmd_run(a):
    systems = SYSTEMS if a.order == "fwd" else list(reversed(SYSTEMS))
    if a.systems:
        systems = [s for s in systems if s in a.systems]
    reps = [run_systems(systems, a.alpha, a.attack, a.seed, a.rounds) for _ in range(a.repeat)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"cell": vars(a), "reps": reps}, open(a.out, "w"), indent=1)


def cmd_check(a):
    runs = []
    for path in a.files:
        d = json.load(open(path))
        runs += [(f"{os.path.basename(path)}#{i}", r) for i, r in enumerate(d["reps"])]
    ref_name, ref = runs[0]
    ok = True
    for name in ref:
        worst = 0.0
        for other_name, other in runs[1:]:
            if name in other:
                worst = max(worst, max(abs(x - y) for x, y in zip(ref[name], other[name])))
        status = "OK" if worst == 0.0 else "NON-DETERMINISTIC"
        ok &= worst == 0.0
        print(f"{name:14s} max|Δ| = {worst:.6f}  {status}")
    print("DETERMINISM", "PASSED" if ok else "FAILED", f"({len(runs)} runs compared against {ref_name})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--order", choices=["fwd", "rev"], default="fwd")
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--alpha", type=float, default=0.1)
    r.add_argument("--attack", default="sign_flipping")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--systems", nargs="*")
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    a = p.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_check(a)


if __name__ == "__main__":
    main()
