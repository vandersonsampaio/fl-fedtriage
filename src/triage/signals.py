"""Per-client signals. Every signal is oriented so that a LOWER value is more suspicious."""

from typing import List, Sequence

import numpy as np

INVERSION = "inversion"
SERVER_COSINE = "server_cosine"
MEDIAN_COSINE = "median_cosine"
SIGNALS = (INVERSION, SERVER_COSINE, MEDIAN_COSINE)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def server_cosine(updates: Sequence[np.ndarray], server_update: np.ndarray) -> np.ndarray:
    """Cosine of each client update to the update the server trains on its root dataset."""
    return np.array([cosine(u, server_update) for u in updates])


def median_cosine(updates: Sequence[np.ndarray]) -> np.ndarray:
    """Cosine of each client update to the coordinate-wise median of the round's updates."""
    med = np.median(np.stack(updates), axis=0)
    return np.array([cosine(u, med) for u in updates])


def inversion_score(
    global_params: np.ndarray,
    updates: Sequence[np.ndarray],
    local_lr: float,
    n_features: int,
    n_classes: int,
    num_images: int = 16,
    max_iters: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """S_R: final cosine between each client's update and the gradient of a batch of dummy
    samples optimized to reproduce it (gradient inversion on the logistic model)."""
    from src.defense.adaaggrl_agent import reconstruct_client_distribution

    K = 1 if n_classes == 2 else n_classes
    scores: List[float] = []
    for u in updates:
        theta = global_params + u
        W = theta[: n_features * K].reshape(n_features, K).astype(np.float32)
        b = theta[n_features * K:].astype(np.float32)
        _, s_r = reconstruct_client_distribution(
            W, b, u, local_lr, n_features, K, num_images=num_images, max_iters=max_iters, seed=seed,
        )
        scores.append(s_r)
    return np.array(scores)
