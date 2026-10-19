# Preregistration: confirmation of TRIAGE (`tri_all`)

**Status:** frozen on 2026-10-19 09:06:20 -0300, before any run on seeds 62–71. The SHA-256 of this file and of the code it relies on is in `PREREGISTRATION.sha256`.

TRIAGE (*Threshold-based Robust Inclusion via Agreement of Gradient Evidence*) decides, for each client and round, whether to include its update. The included updates are averaged with uniform weights. The configuration below was selected on the development seeds (72–81). This document fixes how it is confirmed on seeds that have never been used.

## 1. Frozen configuration

- **System:** `tri_all`. It combines two signals: the inversion score S_R and the cosine with the server update computed on the root dataset. A client is excluded only when both signals flag it. The threshold is b = 0.5 on the per-round min-max scale, and the memory uses λ = 2.
- **Weights of the included clients:** uniform. In the development ablation (`results/ablation_h300`), sample-size weights lost 2.22 p.p. on BloodMNIST at H = 250, and soft weights gained 0.23 p.p. with only 2 of 5 seeds at +1 p.p. or more, so neither met the adoption criterion.
- **No second stage.** The trimmed mean, clustering and norm clipping second stages all failed their criteria during development (`results/backdoor`, `results/clipping`, `results/clipping_h300`).
- **Root, two scenarios:**
  - **canonical (primary):** 100 samples from the server validation set, drawn with the seed;
  - **`@r20` (secondary):** 20 samples from the server validation set, drawn with the seed. It stands for a small, representative root that is public or consented and contains no patient data. In development (`results/root_cost`), `tri_all` lost 0.95 p.p. on BloodMNIST and 0.90 p.p. on MNIST with this root.
- **Model and federation:** logistic model, 10 clients, with clients 0 and 1 Byzantine.

## 2. Datasets, cells and horizons

- **BloodMNIST (primary dataset):** 19 cells, which are 7 attacks × α ∈ {0.5, 0.1, 0.05}, without `fltrust_aligned` at α 0.1 and 0.05.
  - **H = 250**, where the curves level off in `results/horizon300`.
  - Rounds 150 and 300 are reported as secondary.
  - This is the primary dataset because it is where development showed the gain: +3.6 p.p. over the best single signal, against +0.7 p.p. on MNIST.
- **MNIST (secondary dataset):** the same 19 cells at **H = 150**, where the curves are already stable.
- **Seeds:** **62–71**, ten seeds never used before.

## 3. Systems

| role | systems |
|---|---|
| candidate | `tri_all` |
| best single signal (main reference) | `cosserver_only`, `cos_bin`, `sr_bin`; the best is chosen on seeds 62–71 by overall mean |
| fixed rules | `fltrust`, `fedavg`, `trimmed_mean`, `median`, `clustering`, `krum` |
| no root | `flame` |
| ceilings | `oracle_uniform`, `ceiling_fedavg10` |

- **Root scenario `@r20`:** only the systems that use the root are run again (`tri_all`, `cosserver_only`, `cos_bin` and `fltrust`). The others (`sr_bin`, `flame`, the root-free rules and the ceilings) do not depend on the root and are shared by both scenarios.
- **Learned defenses are not compared here.** These are GRADF v1, AdaAggRL (own and official implementations), FedStrategist, TARS and AdaBFL. In the earlier experiments of fl-gradf, none of them beat a fixed action or a random selection, so the confirmatory comparison is against the fixed rules and the single signals. The learned defenses are compared later, on the same seeds and with the same analysis, under a preregistration of their own.

## 4. Hypotheses

**Primary (BloodMNIST, H = 250, canonical root):**
- **H1:** `tri_all` > best single signal.
- **H2:** `tri_all` > `fltrust`.
- **H3:** `tri_all` > `flame`.

**Secondary (BloodMNIST, H = 250, root `@r20`):**
- **H1′:** `tri_all@r20` > best single signal on the same root. The best is chosen among `cosserver_only@r20`, `cos_bin@r20` and `sr_bin` on seeds 62–71 by overall mean.
- **H2′:** `tri_all@r20` > `fltrust@r20`.
- **H3′:** `tri_all@r20` > `flame`.

**Secondary, other:**
- **H4 (MNIST, H = 150, canonical root):** `tri_all` is not inferior to the best single signal, with an equivalence margin of ±1 p.p.

**Descriptive, no test:**
- **Root cost:** the loss of each root-dependent system with `@r20` against the same system with the canonical root, on both datasets.
- **Worst case:** the largest per-cell loss against the best fixed rule of each cell.

## 5. Analysis

- **Unit:** the mean over the 19 cells for each seed, paired by cell and seed.
- **Test:** one-sided paired Wilcoxon test over the 10 per-seed means, α = 0.05.
- **Multiplicity:** Holm over H1–H3, and a separate Holm over H1′–H3′.
- **Non-inferiority (H4):** TOST with a ±1 p.p. margin over the per-seed means.
- **Per cell (descriptive):** a cell counts as a win or a loss when the difference is beyond 2 standard errors, and as a tie otherwise, as in development.
- **Script:** `scripts.triage_compare` with `--seeds 62 63 64 65 66 67 68 69 70 71` and `--horizons`.

## 6. Decision rule

- **Confirmed** if H1 and H2 are significant after Holm and H4 holds.
- H1′–H3′ do not enter the decision. They are reported, and a failure is added to the privacy limitations.
- **If not confirmed:** one return to development is allowed, with a new preregistration.

## 7. Limitations stated in advance

- **Trigger backdoor:** no variant stops it (`results/backdoor`, `results/clipping`, `results/flame`).
- **Label flipping:** the attacker looks like an honest client, and `tri_all` loses to the best rule of the cell.
- **No scale control:** `tri_all` collapses on sign flipping at α = 0.05 after round 200 (`results/collapse_diag`). Clipping prevents the collapse but failed its adoption criterion (`results/clipping_h300`).
- **Inversion:** the inversion score only applies to the logistic model, and it does not pay off on MNIST (`results/flame`).
- **Root representativeness:** the root must be representative, with about 20 samples or more and not drawn from a single client (`results/root_cost`).
- **What `@r20` does not test:** it comes from the same distribution as the clients. It does not test a public root from another distribution or a synthetic one, and it applies no differential privacy to the server update.

## 8. Termination clause

- There is a single run.
- After any result on seeds 62–71 has been seen, nothing is adjusted: not the configuration, the threshold, the horizon or the set of systems.
- Jobs that fail from an infrastructure error are rerun with the same configuration, and each rerun is logged.

## 9. Estimated cost

- **Canonical root:** about 13 systems × 190 jobs (19 cells × 10 seeds) on each dataset, BloodMNIST at H = 250 and MNIST at H = 150. That is about 4,900 jobs and 55,000 CPU minutes, or about 65 h with 14 parallel jobs.
- **Root `@r20`:** 4 systems × 190 jobs × 2 datasets = 1,520 more jobs. They are dominated by `tri_all`, because of the inversion. That is about 22,000 CPU minutes, or about 26 h more.
- **Total:** about 6,400 jobs and about 91 h (3.8 days) with 14 parallel jobs.
