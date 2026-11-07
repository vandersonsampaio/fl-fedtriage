# Result — determinism and parity with fl-gradf

**Date:** 2026-10-10.
**Acceptance criterion:** the same seed gives the same run, |Δ| = 0 on every round, for every baseline system.

## 1. Diagnosis (before the fix)

`scripts/determinism_check.py` runs 13 systems on one cell and records the global accuracy of every round: 6 fixed rules, Random, Oracle, GRADF v1, FedStrategist, AdaAggRL, TARS and AdaBFL. The same seed was run three times: twice in one process, and once in a fresh process with the systems in reverse order.

Cell: MNIST, α = 0.1, `sign_flipping`, seed 42, 5 rounds.

| system | max \|Δ\| before | after |
|---|---|---|
| fedavg / fltrust / median / trimmed_mean / krum / clustering | 0.0023 – 0.0568 | **0** |
| random / oracle | 0.0124 / 0.0236 | **0** |
| **gradf** | **0.1848** | **0** |
| fedstrategist / tars / adabfl | 0.0079 / 0.0070 / 0.0081 | **0** |
| adaaggrl | 0 | 0 |

Files: `before/`, `fwd.json`, `rev.json`, `check.txt` (after).

**Causes found:**
1. **Unseeded global RNG.** Local training (`_LogisticModel.fit`, mini-batch shuffling), the root carve (`_carve_root`), DP noise and the Keras networks (initialization, shuffling) use the global Python/NumPy/TensorFlow RNGs. No learner seeded them. Only the AdaAggRL extractor called `keras.utils.set_random_seed(seed)`, which is why AdaAggRL was the one deterministic system, and why every other system depended on what had run before it in the process.
2. **Wall-clock time in GRADF's DQN state.** The 5-dimensional state includes the measured round latency (`time.perf_counter`), both in offline pretraining (`generate_selector_experiences`) and in online learning. It differs on every run. After fix 1, GRADF still diverged inside one process (the selector's Q-values went from −10 to +686 between two identical pretrainings).

## 2. Fix

- `FederatedLearner.__init__(seed=...)` calls `seed_everything(seed)`, which seeds Python, NumPy and TensorFlow/Keras. `AttackedFederatedLearner` passes its `seed` through, so every seeded learner (all experiments pass one) starts from a known RNG state.
- `RLDefenseSelector.LATENCY_FEATURE = "constant"` (default): the latency feature of the state is 0.0. The state stays 5-dimensional, so the network is unchanged. `FEDTRIAGE_GRADF_LATENCY=wallclock` restores the published behavior.
- Tests:
  - `test_build_state_constant_latency_is_deterministic` is new;
  - two tests that implicitly relied on wall-clock variation were rewritten to check what they meant to check;
  - suite: 96 passed.

**Consequence for GRADF v1 as a baseline:** it now runs without the latency feature. This is a declared deviation from the published GRADF v1, required for determinism. The reward never used latency; only the state did.

## 3. Result

- **Determinism PASSED** on two cells:
  - α = 0.1, `sign_flipping`, seed 42 (`check.txt`);
  - α = 0.05, `label_flipping`, seed 43 (`lf_check.txt`).
- In both, all 13 systems had |Δ| = 0 on every round, across two runs in the same process and one fresh process in reverse order.

## 4. Parity with fl-gradf

Three B2.6 jobs (21 cells each, 15 rounds) were re-run here with the deterministic code and compared with the CSVs published in fl-gradf (`results/b26_decomposicao/raw/`, tag `experiments-2.0.0`). Script: `scripts/parity_fl_gradf.py`; output: `parity_b26.txt`, `parity_raw/`.

| job | cells | max \|Δ\| |
|---|---|---|
| `cosserver_only`, seed 52 | 21/21 | **0** |
| `sr_bin`, seed 52 | 21/21 | **0** |
| `sr_b025`, seed 53 | 21/21 | **0** |

→ **PARITY PASSED.** The AdaAggRL-skeleton variants, the direct basis of TRIAGE, reproduce fl-gradf exactly with the new code. The fix does not change their trajectory, because AdaAggRL already reseeded the global state in its own constructor, after `FederatedLearner.__init__`.

**What does not reproduce fl-gradf exactly, by design:** the systems that were non-deterministic there (rules, GRADF, FedStrategist, TARS, AdaBFL, Random, Oracle). Their fl-gradf numbers were one draw from an order-dependent distribution. From now on they are re-measured with the deterministic code whenever used as baselines.

## 5. Limits

- Determinism was checked on 2 cells × 5 rounds with the logistic model, not on the CNN path;
- Parity was checked on 3 of B2.6's 80 jobs.
