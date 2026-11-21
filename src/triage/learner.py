from typing import Dict, List, Optional, Sequence

import numpy as np

from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
import src.defense.aggregation_methods  # noqa: F401  (registers the robust rules)
from src.fl.federated_learner import RoundResult, _STRATEGIES
from src.triage.rule import TriageRule
from src.triage.signals import (
    INVERSION, MEDIAN_COSINE, NEG_MEDIAN_DISTANCE, NEG_NORM, SERVER_COSINE, SIGNALS, TEMPORAL_COSINE,
    inversion_score, median_cosine, neg_median_distance, neg_norm, server_cosine, temporal_cosine,
)


def cap_included_norms(updates: List[np.ndarray], included: np.ndarray, k: float):
    """Scales down each included update whose norm exceeds k times the median norm of the included
    updates. Returns the updates (unscaled ones are the same objects) and the scale factors."""
    norms = np.array([np.linalg.norm(u) for u in updates])
    idx = np.flatnonzero(included)
    scales = np.ones(len(updates))
    if len(idx) == 0:
        return list(updates), scales
    bound = k * float(np.median(norms[idx]))
    out = list(updates)
    for i in idx:
        if norms[i] > bound:
            scales[i] = bound / norms[i]
            out[i] = updates[i] * scales[i]
    return out, scales


class TriageLearner(AttackedFederatedLearner):
    """Aggregates the clients' full parameters (global + update) with the weights of a
    `TriageRule`. `log` holds one record per client and round (`Decision.records`, plus
    `is_byz` and the raw value of every signal in `observe`, which are logged but not used).

    With `second_stage` (a registered aggregation rule such as "trimmed_mean" or "clustering"),
    the updates of the included clients are aggregated by that rule instead of by the weighted
    mean; if the rule includes no client, every client goes to the second stage.

    With `norm_cap` (a factor k), the update of every included client whose L2 norm exceeds k
    times the median norm of the included clients in the round is scaled down to that bound before
    the weighted mean; nothing else changes, and rounds without such a client are bit-identical to
    the run without `norm_cap`. The log then has a `cap_scale` column (1.0 when not scaled).

    The server update used by `server_cosine` is trained on the root dataset with the global
    NumPy RNG state saved and restored, so computing it does not change the clients' training.
    """

    def __init__(self, *args, rule: TriageRule, num_images: int = 16, max_iters: int = 30,
                 record: bool = True, observe: Sequence[str] = (), second_stage: Optional[str] = None,
                 norm_cap: Optional[float] = None, **kwargs):
        kwargs.setdefault("aggregation", "fedavg")
        super().__init__(*args, **kwargs)
        unknown = set(observe) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signals in observe: {sorted(unknown)}")
        self.signals = list(rule.signals) + [s for s in observe if s not in rule.signals]
        if INVERSION in self.signals and self.model_type != "logistic":
            raise ValueError("the inversion signal needs model_type='logistic'")
        if second_stage is not None and second_stage not in _STRATEGIES:
            raise ValueError(f"unknown second_stage {second_stage!r}")
        if norm_cap is not None and (second_stage is not None or norm_cap <= 0):
            raise ValueError("norm_cap must be positive and cannot be combined with second_stage")
        self.rule = rule
        self.second_stage = second_stage
        self.norm_cap = norm_cap
        self.num_images = num_images
        self.max_iters = max_iters
        self.record = record
        self.log: List[Dict] = []
        self._h: Optional[np.ndarray] = None
        self._previous: Dict[str, np.ndarray] = {}

    def _server_update(self, participants, root_data) -> np.ndarray:
        rng_state = np.random.get_state()
        root = root_data or self._carve_root(participants[0])
        update = self._fit_server(self._make_model(participants[0].n_features), root, participants)
        np.random.set_state(rng_state)
        return update

    def _signals(self, round_num, participants, root_data, updates) -> Dict[str, np.ndarray]:
        raw: Dict[str, np.ndarray] = {}
        if INVERSION in self.signals:
            raw[INVERSION] = inversion_score(
                self._global_params, updates, self.local_lr, participants[0].n_features,
                self.n_classes, num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
            )
        if SERVER_COSINE in self.signals:
            raw[SERVER_COSINE] = server_cosine(updates, self._server_update(participants, root_data))
        if MEDIAN_COSINE in self.signals:
            raw[MEDIAN_COSINE] = median_cosine(updates)
        if NEG_NORM in self.signals:
            raw[NEG_NORM] = neg_norm(updates)
        if NEG_MEDIAN_DISTANCE in self.signals:
            raw[NEG_MEDIAN_DISTANCE] = neg_median_distance(updates)
        if TEMPORAL_COSINE in self.signals:
            raw[TEMPORAL_COSINE] = temporal_cosine(updates, [self._previous.get(p.id) for p in participants])
        self._previous = {p.id: u for p, u in zip(participants, updates)}
        return raw

    def _run_round(self, round_num, participants, root_data):
        active = self._is_active(round_num)
        updates, is_byz = compute_param_updates_auto(self, participants, active, root_data)
        raw = self._signals(round_num, participants, root_data, updates)
        decision = self.rule.decide(raw, self._h, sample_sizes=[p.n_train for p in participants])
        self._h = decision.h

        scales = np.ones(len(updates))
        if self.second_stage is None:
            if self.norm_cap is not None and not decision.fallback:
                updates, scales = cap_included_norms(updates, decision.included, self.norm_cap)
            theta_list = [self._global_params + u for u in updates]
            self._global_params = sum(w * th for w, th in zip(decision.weights, theta_list))
        else:
            kept = [u for u, inc in zip(updates, decision.included) if inc and not decision.fallback] or updates
            delta, _ = _STRATEGIES[self.second_stage]().aggregate(kept)
            self._global_params = self._global_params + delta
        if self.record:
            records = decision.records(round_num, [p.id for p in participants])
            for i, row in enumerate(records):
                row["is_byz"] = bool(is_byz[i])
                if self.norm_cap is not None:
                    row["cap_scale"] = float(scales[i])
                for s in self.signals:
                    if s not in row:
                        row[s] = float(raw[s][i])
            self.log.extend(records)

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc,
                           trust_scores=None, n_accepted=int(decision.included.sum()))
