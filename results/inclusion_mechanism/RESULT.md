# Result — inclusion log and mechanism of `tri_all` (block B1)

**Date:** 2026-11-17. **Status: exploratory and descriptive, on development seeds 72–76; it confirms nothing.** Plan and reading rule: `PLAN.md`. 760/760 jobs, no traceback. 124 jobs ran via the PC dispatcher on 2026-11-04 (`dispatch.log`), the rest on two Azure VMs, 2026-11-11 to 2026-11-16 (`dispatch_vm2.log`, `dispatch_vm3.log`; each half of the job list on one VM). Files were copied to the PC with checksums checked (2280 files; the ones present on both sides were identical). Analysis: `scripts/analyze_inclusion_mechanism.py`, output in `analysis/`.

**In short:** by the frozen rule, the mechanism claim is **"agreement as protection against false positives"**. On BloodMNIST, 97% of the gross flows in `tri_all`'s favor against the best single signal are honest clients it keeps and the signal drops. `tri_all` excludes *fewer* Byzantine clients than any single signal, not more.

## Accuracy at the horizon

Per seed, mean over the 19 cells, CI95 with t, 5 seeds, p.p. Accuracy is not pooled with earlier TRIAGE runs (`PLAN.md`, "Not bit-identical").

| dataset | H | best single signal | `tri_all` − best | `tri_all` − `sr_bin` |
|---|---|---|---|---|
| BloodMNIST | 250 | `cos_bin` (62.15) | +2.76 (+1.46; +4.07) | +8.62 (+7.61; +9.62) |
| BloodMNIST | 300 | `cosserver_only` (62.47) | +2.12 (−1.11; +5.34) | +7.78 (+7.14; +8.42) |
| MNIST | 150 | `cos_bin` (87.71) | +0.60 (−0.74; +1.95) | +2.82 (+2.16; +3.47) |

`tri_all` means: BloodMNIST 64.92 (H 250) and 64.58 (H 300), MNIST 88.31. By α on BloodMNIST at H = 250, `tri_all` − best is +7.88 (0.05), −0.46 (0.10) and +1.14 (0.50).

## Inclusion rates

All rounds and cells. FN = Byzantine clients included; FP = honest clients excluded.

| dataset | system | FN | FP |
|---|---|---|---|
| BloodMNIST | `tri_all` | 0.534 | **0.072** |
| BloodMNIST | `cos_bin` | 0.262 | 0.438 |
| BloodMNIST | `cosserver_only` | 0.255 | 0.429 |
| BloodMNIST | `sr_bin` | 0.426 | 0.222 |
| MNIST | `tri_all` | 0.472 | **0.092** |
| MNIST | `cos_bin` | 0.273 | 0.425 |
| MNIST | `cosserver_only` | 0.274 | 0.408 |
| MNIST | `sr_bin` | 0.318 | 0.212 |

Per attack, α and window: `analysis/rates.csv`.

## Decomposition (reading rule)

The runs are paired by (α, attack, seed, round, client) over rounds 1..H. `tri_all` against the best single signal:

| dataset | H | vs | honest preserved | Byzantine excluded | honest lost | Byzantine let in | share of honest preserved |
|---|---|---|---|---|---|---|---|
| BloodMNIST | 250 | `cos_bin` | 73,239 | 2,215 | 3,966 | 15,034 | **0.971** |
| BloodMNIST | 300 | `cosserver_only` | 86,481 | 2,574 | 5,045 | 18,507 | **0.971** |
| MNIST | 150 | `cos_bin` | 42,270 | 2,090 | 4,274 | 7,783 | 0.953 |

- **Rule:** share > 0.5 on BloodMNIST at both horizons, so the verdict is "agreement as protection against false positives".
- **Stability:** the share ranges from 0.966 to 0.974 across seeds. It ranges from 0.949 (`label_flipping`) to 0.998 (`fltrust_aligned`) across attacks, and from 0.966 to 0.975 across α. It is 0.971 when only the cells where `tri_all` beat the signal are kept.
- **Net flows:** `tri_all` lets in more Byzantine clients than it excludes. The net count of Byzantine excluded is −12,819 against `cos_bin` at H = 250. All of the gain is on the honest side.

## Within `tri_all`'s own run

`tri_all` excludes only when both signals flag (`combine="all"`). Clients flagged by one signal and still included, because the other signal did not flag them (BloodMNIST, H = 250):

| flag | honest flagged, included | Byzantine flagged, included |
|---|---|---|
| `server_cosine` | 85% of 92,686 | 31% of 32,427 |
| `inversion` | 53% of 29,232 | 30% of 31,695 |

The second signal overrides the first far more often on honest clients than on Byzantine ones. This holds on MNIST too (80% against 29% for `server_cosine`).

## Limits

- **The rule's outcome is close to structural.** With `combine="all"`, `tri_all` includes every client that either signal alone would include on the same update. So, against a single signal, "Byzantine excluded" can only come from differences between trajectories. The rule mostly measures whether the gain sits with FP or FN, and here the answer is not close.
- **Cross-run pairing:** the two systems follow different global models after round 1, so the flows compare different updates of the same client. They describe; they do not attribute causally.
- **Counts are not weighted** by effect on the model. The Byzantine clients `tri_all` lets in may do little harm. Their median norm relative to honest clients is 1.74 on BloodMNIST, against 1.35 for `cos_bin`. Under `gaussian_noise`, included Byzantine updates have about 800× the honest norm in every system. Why that does not wreck accuracy was not investigated.
- **At H = 300 and on MNIST, `tri_all` − best is not shown to be above zero.** The decomposition still describes how `tri_all` differs, but there the difference is not a demonstrated gain.
- **Five development seeds.**
