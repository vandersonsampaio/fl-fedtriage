# Combining the inversion score with the server cosine (MNIST)

Logistic model, 10 clients (0 and 1 Byzantine), the 19 attack cells, H = 150, seeds 72–76. All
TRIAGE configurations use b = 0.5 (relative-max threshold), uniform weights and lam = 2. The
single-signal systems (`sr_bin`, `cosserver_only`) and `fltrust`, `sr_b025` come from
`results/c0b_reference/`; `cos_bin` and the candidates were run here.

```
venv/bin/python -m scripts.triage_compare --dataset mnist --raw results/f2_agreement/raw \
    --candidates tri_all tri_conf tri_any tri_rank cos_temp_any cheap_any \
    --single sr_bin cosserver_only --refs cos_bin fltrust sr_b025 --out results/f2_agreement
```

| candidate | signals | combiner | accuracy | vs best single (`cosserver_only`, 0.877) | vs per-cell oracle (0.889) | wins / ties / losses |
|---|---|---|---:|---:|---:|---|
| `tri_all` | inversion, server_cosine | all | 0.883 | +0.65 | −0.56 | 5 / 11 / 3 |
| `tri_conf` | inversion, server_cosine | confidence | 0.882 | +0.52 | −0.68 | 3 / 14 / 2 |
| `tri_any` | inversion, server_cosine | any | 0.868 | −0.84 | −2.04 | 1 / 14 / 4 |
| `tri_rank` | inversion, server_cosine | min_rank | 0.852 | −2.43 | −3.64 | 1 / 11 / 7 |
| `cheap_any` | neg_median_distance, temporal_cosine | any | 0.840 | −3.67 | −4.88 | 3 / 6 / 10 |
| `cos_temp_any` | server_cosine, temporal_cosine | any | 0.837 | −3.98 | −5.19 | 1 / 8 / 10 |

Differences in percentage points, paired by cell and seed. The per-cell oracle picks, in each
cell, the better of `sr_bin` and `cosserver_only`, and is 1.21 p.p. above the best single signal.
Wins and losses are per-cell differences to `cosserver_only` beyond 2 standard errors.

## Findings

- **No combination beats the per-cell choice of signal on MNIST.** `tri_all` and `tri_conf` are
  tied with the best single signal (Wilcoxon p ≥ 0.31 over 5 seeds) and stay below the oracle.
- **Every combination wins `fltrust_aligned`** (+3.7 to +14.3 p.p.), the cell where the server
  cosine is fooled; the combiners differ in what they lose elsewhere.
- **`any` excludes too many honest clients.** On the `sr_bin` trajectory, the union of the two
  b = 0.5 flags excludes 30–90% of the honest clients depending on the attack.
- **`all` lets through the attacks that only one signal sees.** Under label flipping the server
  cosine flags 97–99% of the Byzantine clients, but the intersection only ~40%, which explains the
  losses of `tri_all` in those cells.
- **Pairing with `temporal_cosine` through `any` fails where that signal is inverted**
  (`cos_temp_any`: label flipping −5 to −17 p.p., sign flipping −2 to −21 p.p.), as predicted by
  the per-attack AUCs in `results/signal_auc/`.
