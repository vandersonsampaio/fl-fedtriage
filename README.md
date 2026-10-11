# fl-fedtriage — TRIAGE: Threshold-based Robust Inclusion via Agreement of Gradient Evidence

TRIAGE is a research project on **Byzantine-robust aggregation for Federated Learning (FL)**, with a focus on medical data. Its starting point is a negative result. An audit of adaptive, RL-based defenses found that their performance comes from a fixed, per-client filtering skeleton, not from learning. TRIAGE builds on that skeleton directly, without RL:

- a **per-client inclusion decision** (include or exclude), instead of soft weighting or the choice of one aggregation rule per round;
- a **threshold** as the main design parameter;
- **agreement between independent pieces of gradient evidence** as the signal.

> **Status:** starting point. This repository holds the framework, baselines and prototypes inherited from [fl-gradf](https://github.com/vandersonsampaio/fl-gradf). The TRIAGE module (`src/triage/`) has its building blocks (signals, threshold, mask, weighting, per-client log) and reproduces the AdaAggRL-skeleton variants exactly; the method's signal-agreement variants are under development (TRIAGE is the thesis' GRADF-v2).

- [Background: what the fl-gradf audit established](#background-what-the-fl-gradf-audit-established)
- [Provenance](#provenance)
- [Contents](#contents)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)

---

## Background: what the fl-gradf audit established

All results below are pre-registered and documented in [fl-gradf `results/`](https://github.com/vandersonsampaio/fl-gradf/tree/experiments-2.0.0/results), with seeds, commands and frozen hashes.

| Finding | Source in fl-gradf |
|---|---|
| Discrete rule selection (GRADF's DQN, FedStrategist) does not beat random selection; continuous per-client weighting (AdaAggRL) captures most of the headroom | tag [`p1.0.0`](https://github.com/vandersonsampaio/fl-gradf/tree/p1.0.0) (paper P1) |
| Removing AdaAggRL's TD3 does not hurt, and slightly improves accuracy; S_R alone is as good as the full signal set | `results/frente1_ablacao_adaaggrl/` |
| In the official AdaAggRL code, TD3 is equivalent to a fixed action, on MNIST and on BloodMNIST; the policy does not move from its initialization | `results/b21_replicacao_oficial/`, `results/b31_medmnist_oficial/` |
| No TD3 configuration (steelman) beats the fixed action | `results/b23_steelman_oficial/`, `results/b23b_steelman_normalizado/` |
| A binary mask is as good as or better than soft weighting; memory does not help on average; **the threshold has the largest effect**; S_R and `cos_server` are complementary | `results/b26_decomposicao/`, `results/a0_analises/` |
| The threshold is causal, but no constant beats the center (MNIST and BloodMNIST under EB) | `results/b24r_limiar_oficial/`, `results/b33_limiar_bloodmnist/` |
| On BloodMNIST, the published mechanism runs in a degraded, periodic-reset regime (~31% accuracy under attack) | `results/b31_medmnist_oficial/`, `results/b33_limiar_bloodmnist/` |

In short: the useful part of adaptive defense is **which clients enter the aggregate, and by what threshold**. TRIAGE asks whether making that decision explicit, from agreement between complementary signals, gives a defense that is simpler, auditable and at least as strong.

---

## Provenance

This repository was created from **[fl-gradf](https://github.com/vandersonsampaio/fl-gradf) at tag `experiments-2.0.0` (commit `9f39371`)**, with a fresh git history.

- **fl-gradf remains the reference** for GRADF v1, the P1 paper (tag `p1.0.0`) and the AdaAggRL audit. Its pre-registrations, results and frozen hashes are not copied here; they can only be verified against fl-gradf's history.
- Only versioned files were imported, and only what the framework needs to run or what TRIAGE builds on (see [Contents](#contents)).
- The imported code is unchanged except for the determinism fix (see below), the `README` files, `.gitignore`, `external/.gitignore`, `config/config.yaml` and `setup.py`. Some docstrings still mention fl-gradf result folders (e.g. `results/b26_decomposicao/`); those refer to fl-gradf.
- **Determinism:** every seeded learner now seeds Python, NumPy and TensorFlow/Keras at construction, and GRADF v1's DQN no longer feeds the measured round time into its state (restorable with `FEDTRIAGE_GRADF_LATENCY=wallclock`). The same seed gives exactly the same run for all 13 baselines (`scripts/determinism_check.py`, `results/determinism/`). Numbers of systems that were non-deterministic in fl-gradf (all except the AdaAggRL-based ones) therefore do not reproduce fl-gradf's exactly.
- Removed after the import, as not needed for TRIAGE: GRADF v1's own experiments (`exp1_cnn_scale`, `exp2`, `exp3`, `exp5`–`exp8`, `runner`) and the MIMIC-III tooling (`data/download_mimic.py`, `data/merge_mimic_tables.py`, `src/utils/hospital_splitter.py`); the thesis replaced MIMIC with eICU-CRD. They remain in fl-gradf.

---

## Contents

| Part | What it is | Origin in fl-gradf |
|---|---|---|
| `src/` | FL framework (NumPy logistic model and Keras CNN), 11+ attacks, 7 aggregation rules (FedAvg, FLTrust, Median, Trimmed Mean, Krum, Clustering, FedProx), detectors, and reproductions of GRADF, TARS, AdaBFL, FedStrategist and AdaAggRL. These are the baselines; `src/experiments/` keeps `exp1`, `exp4`, `exp9` and `exp10`, which they depend on | `src/` (architecture in [`src/README.md`](src/README.md)) |
| `scripts/frente1_ablacao_adaaggrl.py` | AdaAggRL without RL: fixed action, `sr_only`, `cos_server`, `cos_median` variants. **The closest prototype of TRIAGE's signals** | Front 1 ablation |
| `scripts/b26_decomposicao.py` | Decomposition of the AdaAggRL skeleton: memory, binary mask vs. soft weighting, signal combination | B2.6 |
| `scripts/passo2_oficial/` | Harness for the **official AdaAggRL code** (unedited, commit `27b9c18`): runner, fixed/TD3 conditions, reset logging, BloodMNIST adaptation (`bloodmnist_shim.py`, extractor training) | Step 2, B2.1, B3.x |
| `scripts/pretrain_adaaggrl_extractor.py`, `results/models/` | Pretrained MNIST extractor used by the in-house AdaAggRL | P1 |
| `results/tables/` | P1 reference tables read by the prototypes: the fixed-rule grid (`exp9_…_ALL_root100_raw.csv`) and AdaAggRL per seed (`exp10_…_variantb_full_seed*_raw.csv`) | P1 |
| `scripts/janela_execucao.sh` | Optional execution window: suspends a grid's process group on weekdays 7h–18h and resumes it afterwards | audit |
| `data/` | Download and partitioning of MNIST/CIFAR-10 (IID and Dirichlet non-IID) | `data/download_datasets.py` |
| `data/prepare_bloodmnist.py`, `scripts/bloodmnist_sanity.py`, `results/bloodmnist_sanity/` | BloodMNIST partitions in the MNIST layout and the FedAvg sanity check (logistic and CNN) | new |
| `src/triage/`, `scripts/triage_parity.py`, `results/triage_parity/` | TRIAGE building blocks and their parity with the AdaAggRL-skeleton variants of fl-gradf | new |
| `scripts/determinism_check.py`, `scripts/parity_fl_gradf.py`, `results/determinism/` | Same-seed determinism check for all baselines and parity with fl-gradf | new |
| `tests/`, `src/**/tests/` | 117 unit and integration tests | new determinism and TRIAGE tests |

Not imported: GRADF's notebooks, the paper drafts, the audit result folders and their analysis scripts, and the local references. Removed after the import: see [Provenance](#provenance).

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requiriments.txt
python data/download_datasets.py      # MNIST + CIFAR-10 shards under data/processed/
python test_setup.py
pytest                                # 95 tests
```

Experiments are run as modules from the repository root, so that `src.*` imports resolve:

```bash
# fixed-rule grid and selector comparison (P1 protocol: canonical root = 100)
python -m src.experiments.exp9_dominance_grid --seeds 42 43 44 45 46 47 48 49 50 51
python -m src.experiments.exp10_selector_comparison --variant b

# AdaAggRL skeleton without RL (CPU)
CUDA_VISIBLE_DEVICES="" python -m scripts.frente1_ablacao_adaaggrl run --mode sr_only --seeds 42
```

**Official AdaAggRL (GPU, isolated venv):**

```bash
git clone https://github.com/yjEugenia/AdaAggRL.git external/AdaAggRL && git -C external/AdaAggRL checkout 27b9c18
# build external/.venv_adaaggrl with the versions listed in external/.gitignore
external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_oficial.py --attack LMP --condition fixed --seed 100 --rounds 500 --q 0.5 --out_dir results/<experiment>/raw

# BloodMNIST: data/raw/medmnist/bloodmnist.npz from MedMNIST v2 (Zenodo record 10519652), then
external/.venv_adaaggrl/bin/python scripts/passo2_oficial/treinar_extrator_bloodmnist.py
external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b3.py --dataset BloodMNIST --attack EB --condition fixed --seed 100 --out_dir results/<experiment>/raw
```

The inherited scripts default to fl-gradf's output folders; pass `--out_dir` (or edit `OUT`) to write elsewhere. A 500-round official run takes ~1.5 GPU-hours on an RTX 3050; 4–6 processes saturate the GPU.

---

## Repository layout

```
src/            FL framework, attacks, aggregation rules, detectors, reproduced defenses (baselines)
scripts/        AdaAggRL-skeleton prototypes and the official-code harness (scripts/passo2_oficial/)
data/           dataset download and partitioning scripts (data files are not versioned)
results/        outputs; tables/ and models/ hold the inherited P1 references
tests/          integration tests (unit tests live in src/**/tests/)
config/         FL and network hyperparameters
external/       official AdaAggRL clone and its venv (not versioned)
```

## License

See [`LICENSE`](LICENSE).
