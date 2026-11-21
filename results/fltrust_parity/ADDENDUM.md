# Addendum — FLTrust with step parity on the development seeds

**Date:** 2026-11-02, before these runs. **Authorized by the author on 2026-11-02.** Descriptive, on development seeds; it confirms nothing.

## Why

In P1 (`fl-gradf`, branch `p1/segunda-submissao`, `results/p1_g0_probe/`), FLTrust's server update was found to depend on the root size through the number of server steps. FLTrust (Cao et al., NDSS 2021) gives server and clients the same number of local iterations. Here the server trains by epochs over the root, so at root 100 it takes 20 steps against a median per client of about 37–167 on BloodMNIST and about 750–920 on MNIST.

The decision follows the specification, not the numbers: the main version preserves step parity (server steps = median over clients of ceil(n_i / 32) × local epochs, from the partition only). The epoch-based version stays as the sensitivity version. This is not faithful to the paper's R_l, b and β, because clients here train by epochs.

## Question

Are the claims against FLTrust (H2 and H2′ of `results/f3_confirmation/`) at risk under parity?

## Code

`server_steps` in `FederatedLearner` (`native` by default, `median`, or a positive integer) and one `_fit_server` method at every server fit in `src/`, plus `RuleLearner` in `scripts/triage_grid.py`. `--server_steps` in `triage_grid`; non-native outputs get `@st<value>` in the system name. Parity draws `steps` mini-batches from the root with NumPy's global RNG and makes one pass. `TriageLearner` keeps its save and restore of the RNG state around the fit. `scripts/frente1_ablacao_adaaggrl.py` and `scripts/b26_decomposicao.py` keep the epoch-based fit and are not used here.

**Regression before this addendum (8 jobs, 40 rounds, both datasets, both roots, `fltrust` and `tri_all`, including `fltrust_aligned`):** the native path equals `results/f3_confirmation/raw` with max |Δ| = 0 in all 8 (`regression.txt`).

## Runs

`fltrust`, `server_steps = median`, 19 cells, **seeds 72–76**: 380 jobs (`jobs.txt`), 7 parallel jobs of 2 threads, alongside the W1 rerun of `fl-gradf`.

| dataset | root | H | native FLTrust and `tri_all`, already available |
|---|---|---|---|
| BloodMNIST | 100 | 300 | `results/horizon300` |
| BloodMNIST | 20 | 150 | `results/root_cost` |
| MNIST | 100 | 150 | `results/c0b_reference`, `results/f2_agreement` |
| MNIST | 20 | 150 | `results/root_cost` |

**Deviation from `plans/fltrust_server_steps.md`, Sec. 6:** that list named seeds 72–81, at about 1–3 h. Measured costs put 760 jobs at about 84 CPU hours, and the references at these horizons and roots exist on 72–76 only. Seeds 77–81 can be added if the reading is ambiguous.

**Client streams:** `RuleLearner` does not isolate the server fit from NumPy's global RNG, so parity also shifts the clients' mini-batch order relative to native. Differences below include that effect.

## Reading rule

Per seed, mean over the 19 cells, CI95 with t, 5 seeds, at H = 250 and 150 on BloodMNIST root 100, and at H = 150 elsewhere.

Reported for each dataset and root: FLTrust parity − native, `tri_all` − FLTrust native, and `tri_all` − FLTrust parity.

**On BloodMNIST, for each root, a claim is at risk if either holds:**
1. the CI95 of `tri_all` − FLTrust parity is not entirely above zero;
2. the mean of `tri_all` − FLTrust parity is below half of its native value.

**At risk** → step 2 of the list (`tri_all` and the cosine signals under parity), then a decision on a new confirmation on fresh seeds. **Not at risk** → FLTrust's comparisons are reported with both versions, and step 2 is decided by the unmeasured direction question.

**Prediction, stated before the runs:** the effect is smaller on BloodMNIST than on MNIST, since the gap in steps is smaller. At root 20 under parity, g₀ may overfit the root, so a residual root effect is not a pure content effect.

**Limits:** `tri_all` stays native here, although it also reads g₀'s direction. So `tri_all` − FLTrust parity is a screen for risk and not an estimate of the corrected contrast. Seeds 62–71 are not touched.

## Code hashes (sha256)

```
9aabfc279c537d1505e4e49ca8b66435b5e5c3f3e1e268a38ee041c3c5ba6e04  src/fl/federated_learner.py
2c6d5726ae46087dfb9ae52b792a26b1f485d779bf389826d70e62ec5b0cc29b  src/fl/attacked_learner.py
44d66ef80901350ca795a7e25c7a70e338ecc31de1245ec878e7dfc10c848163  src/triage/learner.py
315cb46d2fe1182b8acd1a0ced44c4507d5f99a7afb1916af26e3f47d31c4474  scripts/triage_grid.py
055caba19ba2e7cd1f0d92caad9f2ef9269c793b750ad1095a64a6fe51b5a59b  results/fltrust_parity/jobs.txt
```
