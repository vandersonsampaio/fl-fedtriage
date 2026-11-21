# Result — parity of `scripts/triage_grid.py` with fl-gradf's C0b

**Date:** 2026-10-11.
**Criterion:** |Δ| = 0 against `results/c0b_reference/` (fl-gradf C0b) on the same cell, seed and horizon.

Seed 72, the 19 C0b cells, 15 rounds (C0b's H = 15 point), logistic model. Systems: `fltrust`, `trimmed_mean`, `clustering`, `median`, `cosserver_only` (19 cells each) and `ceiling_fedavg10`, `oracle_fedavg8` (3 α each).

| system | pairs | max \|Δ\| |
|---|---|---|
| ceiling_fedavg10 | 3 | **0** |
| oracle_fedavg8 | 3 | **0** |
| fltrust | 19 | **0** |
| trimmed_mean | 19 | **0** |
| clustering | 19 | **0** |
| median | 19 | **0** |
| cosserver_only | 19 | **0** |

→ **PARITY PASSED** (101/101). Raw per-round accuracy in `raw/mnist/`.

**Consequence:** fl-gradf's C0b ran every system after reseeding the global RNGs, so its numbers are deterministic and equal to what this repository produces. The C0b tables (`results/c0b_reference/`) serve as the MNIST references on seeds 72–81 without re-running them.

**Limit:** H = 15 only; trajectories at H = 50 and 150 are the continuation of the same deterministic runs, but were not re-run here.
