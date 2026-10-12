from typing import Dict, List, Optional, Sequence

import numpy as np

from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
from src.fl.federated_learner import RoundResult
from src.triage.rule import TriageRule
from src.triage.signals import (
    INVERSION, MEDIAN_COSINE, NEG_MEDIAN_DISTANCE, NEG_NORM, SERVER_COSINE, SIGNALS, TEMPORAL_COSINE,
    inversion_score, median_cosine, neg_median_distance, neg_norm, server_cosine, temporal_cosine,
)


class TriageLearner(AttackedFederatedLearner):
    """Aggregates the clients' full parameters (global + update) with the weights of a
    `TriageRule`. `log` holds one record per client and round (`Decision.records`, plus
    `is_byz` and the raw value of every signal in `observe`, which are logged but not used).

    The server update used by `server_cosine` is trained on the root dataset with the global
    NumPy RNG state saved and restored, so computing it does not change the clients' training.
    """

    def __init__(self, *args, rule: TriageRule, num_images: int = 16, max_iters: int = 30,
                 record: bool = True, observe: Sequence[str] = (), **kwargs):
        kwargs.setdefault("aggregation", "fedavg")
        super().__init__(*args, **kwargs)
        unknown = set(observe) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signals in observe: {sorted(unknown)}")
        self.signals = list(rule.signals) + [s for s in observe if s not in rule.signals]
        if INVERSION in self.signals and self.model_type != "logistic":
            raise ValueError("the inversion signal needs model_type='logistic'")
        self.rule = rule
        self.num_images = num_images
        self.max_iters = max_iters
        self.record = record
        self.log: List[Dict] = []
        self._h: Optional[np.ndarray] = None
        self._previous: Dict[str, np.ndarray] = {}

    def _server_update(self, participants, root_data) -> np.ndarray:
        rng_state = np.random.get_state()
        root = root_data or self._carve_root(participants[0])
        update = self._make_model(participants[0].n_features).fit(root["X"], root["y"])
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

        theta_list = [self._global_params + u for u in updates]
        self._global_params = sum(w * th for w, th in zip(decision.weights, theta_list))
        if self.record:
            records = decision.records(round_num, [p.id for p in participants])
            for i, row in enumerate(records):
                row["is_byz"] = bool(is_byz[i])
                for s in self.signals:
                    if s not in row:
                        row[s] = float(raw[s][i])
            self.log.extend(records)

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc,
                           trust_scores=None, n_accepted=int(decision.included.sum()))
