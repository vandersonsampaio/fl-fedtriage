"""Inclusion decision: per-signal normalization and threshold, combination of the per-signal
flags into a binary mask, misbehavior memory and aggregation weights."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.triage.signals import SIGNALS

COMBINERS = ("any", "all")
WEIGHTINGS = ("uniform", "soft", "samples")


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def relative_max_threshold(normalized: np.ndarray, b: float) -> float:
    return float(normalized.max() * b)


@dataclass
class Decision:
    raw: Dict[str, np.ndarray]
    normalized: Dict[str, np.ndarray]
    thresholds: Dict[str, float]
    flags: Dict[str, np.ndarray]
    included: np.ndarray
    weights: np.ndarray
    h: np.ndarray
    fallback: bool

    def reasons(self) -> List[str]:
        """Per client: "fallback" (every weight was zero), "included", or "excluded:<signals>"
        naming the signals that flagged the client."""
        out = []
        for i, inc in enumerate(self.included):
            if self.fallback:
                out.append("fallback")
            elif inc:
                out.append("included")
            else:
                out.append("excluded:" + "+".join(s for s, f in self.flags.items() if f[i]))
        return out

    def records(self, round_num: int, client_ids: Sequence[str]) -> List[Dict]:
        rows = []
        for i, (cid, reason) in enumerate(zip(client_ids, self.reasons())):
            row: Dict = {"round": round_num, "client": cid}
            for s in self.raw:
                row[s] = float(self.raw[s][i])
                row[f"{s}_norm"] = float(self.normalized[s][i])
                row[f"{s}_threshold"] = self.thresholds[s]
                row[f"{s}_flag"] = bool(self.flags[s][i])
            row.update(included=bool(self.included[i]), weight=float(self.weights[i]),
                       h=int(self.h[i]), reason=reason)
            rows.append(row)
        return rows


class TriageRule:
    """Excludes a client when its signals fall at or below their thresholds and weights the rest.

    Each signal is min-max normalized across the round's clients and flags the clients whose
    normalized value is at or below `b` times the round maximum. `combine="any"` excludes a client
    flagged by any signal; `combine="all"` only when every signal flags it.

    Weights of included clients: 1 (`uniform`), the normalized signal (`soft`, single signal
    only) or the client's sample count (`samples`). Each weight is divided by `lam ** h`, where
    the counter `h` grows by 1 on every round the client is excluded and decays by 1 otherwise;
    `lam = 1` disables the memory. If every weight is zero, all clients get the same weight.
    """

    def __init__(self, signals: Sequence[str], b: float = 0.5, combine: str = "any",
                 weighting: str = "uniform", lam: float = 1.0):
        signals = list(signals)
        if not signals or any(s not in SIGNALS for s in signals):
            raise ValueError(f"signals must be a non-empty subset of {SIGNALS}, got {signals}")
        if combine not in COMBINERS:
            raise ValueError(f"combine must be one of {COMBINERS}, got {combine!r}")
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        if weighting == "soft" and len(signals) != 1:
            raise ValueError("weighting='soft' needs exactly one signal")
        if lam < 1.0:
            raise ValueError(f"lam must be >= 1, got {lam}")
        self.signals = signals
        self.b = b
        self.combine = combine
        self.weighting = weighting
        self.lam = lam

    def decide(self, raw: Dict[str, np.ndarray], h: Optional[np.ndarray] = None,
               sample_sizes: Optional[Sequence[int]] = None) -> Decision:
        n = len(raw[self.signals[0]])
        h = np.zeros(n) if h is None else h
        raw = {s: np.asarray(raw[s], dtype=float) for s in self.signals}
        normalized = {s: minmax(raw[s]) for s in self.signals}
        thresholds = {s: relative_max_threshold(normalized[s], self.b) for s in self.signals}
        flags = {s: normalized[s] <= thresholds[s] for s in self.signals}

        stacked = np.stack(list(flags.values()))
        excluded = stacked.any(axis=0) if self.combine == "any" else stacked.all(axis=0)
        new_h = np.where(excluded, h + 1, np.maximum(h - 1, 0))

        if self.weighting == "uniform":
            base = np.ones(n)
        elif self.weighting == "soft":
            base = normalized[self.signals[0]]
        else:
            if sample_sizes is None:
                raise ValueError("weighting='samples' needs sample_sizes")
            base = np.asarray(sample_sizes, dtype=float)
        w = np.where(excluded, 0.0, base) / (self.lam ** new_h)

        total = float(w.sum())
        fallback = total < 1e-8
        weights = np.ones(n) / n if fallback else w / total
        return Decision(raw, normalized, thresholds, flags, ~excluded, weights, new_h, fallback)
