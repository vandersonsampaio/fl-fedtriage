"""Adaptive attacks against the inclusion rule (block C, requirement R5).

The Byzantine clients know the defense: its signals, thresholds and combination rule, the global
model, the server's root update and the other clients' honest updates of the round (white box,
omniscient). Each round they send

    u(λ) = (1 − λ) h_i + λ m_i,

where h_i is the client's own honest update and m_i the malicious update of the base attack, and
pick the largest λ in {1.0, 0.9, ..., 0.0} that passes the chosen test (the same λ for both
colluding clients). λ = 0 sends the honest update and does no damage.

Modes:
  evade     passes the defense's own rule: neither client is excluded. Against the AND rule the
            search finds by itself whichever signal is easier to keep above its threshold.
  srmimic   keeps S_R above its threshold (b · round maximum), whatever the cosine says: the
            "fool only S_R" case.
  cosmimic  keeps the server cosine above its threshold, whatever S_R says.
  creep     evade, then multiplies the crafted update by κ_t, which grows by 5% per round
            (capped at 100). Both signals are scale-invariant, so the client stays included while
            its norm grows: the late-collapse mechanism, made intentional.
  onoff     evade in rounds 1–10, 21–30, ...; honest updates in the other blocks of 10 rounds.

Base attacks (m_i): sign (−h_i), noise (h_i plus the simulator's Gaussian noise, as in
`gaussian_noise`) and label (the update trained on label-flipped data, as in `label_flipping`).
For these, λ < 0.5 with `sign` only shrinks the honest update, so interpolation is weak there.

The `agr` base is the AGR-tailored attack of Shejwalkar & Houmansadr (NDSS 2021) instead of an
interpolation: both clients send μ + γ p, with μ the mean of the honest updates of the round and
p = −(coordinate-wise standard deviation of the honest updates), and γ the largest value in
{32, 16, ..., 0.125, 0} that passes the test. The `lam` log column then holds γ. Honest training
is the same as under `sign_flipping`.
Honest clients train exactly as under the base attack, in the same order and with the same RNG
draws, so their updates match the non-adaptive run of the same base.
"""

from typing import List, Optional, Tuple

import numpy as np

from src.fl.attacked_learner import DATA_ATTACKS
from src.triage.learner import TriageLearner
from src.triage.rule import TriageRule, minmax
from src.triage.signals import INVERSION, SERVER_COSINE, inversion_score, server_cosine

MODES = ("evade", "srmimic", "cosmimic", "creep", "onoff")
BASES = {"sign": "sign_flipping", "noise": "gaussian_noise", "label": "label_flipping", "agr": "sign_flipping"}
LAMBDAS = [round(1.0 - 0.1 * k, 1) for k in range(11)]
GAMMAS = [2.0 ** k for k in range(5, -4, -1)] + [0.0]
CREEP_RATE, CREEP_CAP = 1.05, 100.0
ONOFF_BLOCK = 10


def parse_adaptive(name: str) -> Optional[Tuple[str, str]]:
    """'evade_sign' -> ('evade', 'sign'); None if `name` is not an adaptive attack."""
    mode, _, base = name.partition("_")
    return (mode, base) if mode in MODES and base in BASES else None


ADAPTIVE_ATTACKS = [f"{m}_{b}" for m in MODES for b in BASES]


class AdaptiveTriageLearner(TriageLearner):
    """`TriageLearner` whose Byzantine clients run an adaptive attack (module docstring).
    `crafted` holds, per round, the λ chosen and the scale κ applied."""

    def __init__(self, *args, adaptive: str, **kwargs):
        parsed = parse_adaptive(adaptive)
        if parsed is None:
            raise ValueError(f"unknown adaptive attack {adaptive!r}; expected one of {ADAPTIVE_ATTACKS}")
        self.mode, base = parsed
        self.base = base
        kwargs["attack_type"] = BASES[base]
        super().__init__(*args, **kwargs)
        self.adaptive = adaptive
        self._kappa = 1.0
        self._round = 0
        self._root = None
        self.crafted: List[dict] = []

    def train(self, participants, root_data=None, verbose=True):
        self._root = root_data
        return super().train(participants, root_data=root_data, verbose=verbose)

    def _run_round(self, round_num, participants, root_data):
        self._round = round_num
        return super()._run_round(round_num, participants, root_data)

    # -- updates -----------------------------------------------------------------------------

    def _compute_param_updates(self, participants, active):
        honest, malicious = [], {}
        for i, p in enumerate(participants):
            byz = i in self.byzantine_ids
            if byz and active and self.attack_type in DATA_ATTACKS:
                state = np.random.get_state()
                h = self._make_model(p.n_features).fit(p.X_train, p.y_train)
                np.random.set_state(state)
                malicious[i] = self._make_model(p.n_features).fit(*self._poisoned_data(p))
            else:
                h = self._make_model(p.n_features).fit(p.X_train, p.y_train)
                if byz and active:
                    malicious[i] = self._sim.poison_update(h, self.attack_type)
            honest.append(h)
        is_byz = [i in malicious for i in range(len(participants))]
        if not malicious:
            return honest, is_byz
        if self.mode == "onoff" and ((self._round - 1) // ONOFF_BLOCK) % 2 == 1:
            self.crafted.append({"round": self._round, "lam": 0.0, "kappa": 1.0})
            return honest, is_byz

        lam, updates = self._search(participants, honest, malicious)
        kappa = 1.0
        if self.mode == "creep" and lam > 0:
            kappa = self._kappa
            updates = [u * kappa if i in malicious else u for i, u in enumerate(updates)]
            self._kappa = min(self._kappa * CREEP_RATE, CREEP_CAP)
        self.crafted.append({"round": self._round, "lam": lam, "kappa": kappa})
        return updates, is_byz

    def _search(self, participants, honest, malicious):
        byz = sorted(malicious)
        if self.base == "agr":
            ref = [u for i, u in enumerate(honest) if i not in malicious]
            mu, p = np.mean(ref, axis=0), -np.std(ref, axis=0)
            grid = [(g, {i: mu + g * p for i in byz}) for g in GAMMAS]
        else:
            grid = [(lam, {i: (1.0 - lam) * honest[i] + lam * malicious[i] for i in byz}) for lam in LAMBDAS]
        signals = self._attacker_signals()
        self._g0 = self._server_update(participants, self._root) if SERVER_COSINE in signals else None
        raw_honest = self._raw(participants, honest, signals)
        for lam, crafted in grid:
            cand = list(honest)
            for i in byz:
                cand[i] = crafted[i]
            if lam == 0.0 or self._passes(participants, cand, byz, raw_honest, signals):
                return lam, cand
        return 0.0, list(honest)

    # -- the attacker's copy of the defense ----------------------------------------------------

    def _attacker_signals(self) -> List[str]:
        if self.mode == "srmimic":
            return [INVERSION]
        if self.mode == "cosmimic":
            return [SERVER_COSINE]
        return list(self.rule.signals)

    def _raw(self, participants, updates, signals, only=None):
        idx = range(len(updates)) if only is None else only
        sub = [updates[i] for i in idx]
        raw = {}
        if INVERSION in signals:
            raw[INVERSION] = inversion_score(
                self._global_params, sub, self.local_lr, participants[0].n_features, self.n_classes,
                num_images=self.num_images, max_iters=self.max_iters, seed=self._round)
        if SERVER_COSINE in signals:
            raw[SERVER_COSINE] = server_cosine(sub, self._g0)
        unknown = set(signals) - {INVERSION, SERVER_COSINE}
        if unknown:
            raise ValueError(f"adaptive attacks support only S_R and the server cosine, got {sorted(unknown)}")
        return raw

    def _passes(self, participants, cand, byz, raw_honest, signals) -> bool:
        raw_byz = self._raw(participants, cand, signals, only=byz)
        raw = {s: raw_honest[s].copy() for s in signals}
        for s in signals:
            raw[s][byz] = raw_byz[s]
        if self.mode in ("srmimic", "cosmimic"):
            (s,) = signals
            norm = minmax(raw[s])
            return bool((norm[byz] > self.rule.b * norm.max()).all())
        decision = self.rule.decide(raw, self._h)
        return bool(decision.included[byz].all())
