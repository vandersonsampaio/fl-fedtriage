# Result — TRIAGE parity with the AdaAggRL-skeleton variants

**Date:** 2026-10-11.
**Criterion:** `TriageLearner` with the `TriageRule` configuration of each variant reproduces the B2.6 CSVs published in fl-gradf (`results/b26_decomposicao/raw/`, tag `experiments-2.0.0`) with |Δ| = 0 on every cell.

## Setup

Grid of `scripts/b26_decomposicao.py`: MNIST, α ∈ {0.5, 0.1, 0.05}, 7 attacks (4 informed, 3 blind), clients 0 and 1 Byzantine, 15 rounds, logistic model. Script: `scripts/triage_parity.py`; comparison: `scripts/parity_fl_gradf.py`.

| variant | `TriageRule` |
|---|---|
| `sr_only` | `signals=["inversion"], b=0.5, weighting="soft", lam=2` |
| `sr_bin` | `signals=["inversion"], b=0.5, weighting="uniform", lam=2` |
| `sr_b025` | `signals=["inversion"], b=0.25, weighting="soft", lam=2` |
| `cosserver_only` | `signals=["server_cosine"], b=0.5, weighting="soft", lam=2` |

## Result

| job | cells | max \|Δ\| |
|---|---|---|
| `sr_only`, seed 52 | 21/21 | **0** |
| `sr_bin`, seed 52 | 21/21 | **0** |
| `sr_b025`, seed 53 | 21/21 | **0** |
| `cosserver_only`, seed 52 | 21/21 | **0** |

→ **PARITY PASSED** (`parity.txt`, CSVs in `raw/`). The four variants are now special cases of `TriageRule`, so the skeleton references and the TRIAGE variants share one implementation.

## Limits

- One seed per variant (84 cells in total), out of 10 seeds per variant in B2.6.
- `combine="all"`, `weighting="samples"` and `median_cosine` have no fl-gradf counterpart and are covered only by unit tests.
