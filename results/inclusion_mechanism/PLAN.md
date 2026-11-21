# Plan — inclusion log and mechanism of `tri_all` (block B1–B3)

**Date:** 2026-11-04. **Status:** exploratory and descriptive, written before any run. It confirms nothing, and `tri_all`'s configuration does not change.

**Question:** does `tri_all`'s gain come from detecting attackers better, or from excluding fewer honest clients?

## Grid

- Systems: `tri_all`, `cos_bin`, `cosserver_only`, `sr_bin`, with `--log`.
- BloodMNIST at 300 rounds and MNIST at 150; the 19 attack cells; development seeds 72–76.
- 760 jobs. Server steps `native`, as in the confirmation.

## Metrics

Per attack, per α and per window of rounds:

- rate of Byzantine clients included (false negative);
- rate of honest clients excluded (false positive);
- ratio between the norm of the included Byzantine clients and the norm of the honest ones.

Descriptive decomposition of `tri_all`'s gain over the best single signal: honest clients preserved (included by `tri_all`, excluded by the signal) and Byzantine clients excluded (excluded by `tri_all`, included by the signal).

## Reading rule, fixed before the runs

If more than half of the gain on BloodMNIST comes from honest clients preserved, the mechanism claim becomes "agreement as protection against false positives". Otherwise, "agreement as complementary detection".

## Execution

- Runs on two cloud VMs of the same type (Azure D4as_v7, AMD EPYC 9V45), in the idle capacity left by the P1 campaign, 4 processes per VM, 1 thread per process, `OPENBLAS_CORETYPE=Haswell`, `ONEDNN_MAX_CPU_ISA=AVX2`. It stops at job boundaries when the campaign needs the VMs.
- **Not bit-identical to earlier TRIAGE runs.** A 10-round `tri_all` job gives different accuracy on the VM and on the PC, and on the PC with 1 and with 2 threads (round 1: 0.0941 on the VM; 0.1272 and 0.1055 on the PC, the last one matching `results/horizon300`). Comparisons are therefore made only inside this grid, and accuracy here is not pooled with `results/horizon300` or `results/f3_confirmation`.
