# MAD threshold for the server cosine (MNIST)

`cos_mad2` and `cos_mad3` are `cos_bin` (server cosine, uniform weights, lam = 2) with the
relative-max threshold replaced by median − k · 1.4826 · MAD of the normalized signal, k ∈ {2, 3}.
Same grid as `results/f2_agreement/`: 19 cells, H = 150, seeds 72–76.

```
venv/bin/python -m scripts.triage_compare --dataset mnist --raw results/f2_mad/raw results/f2_agreement/raw \
    --candidates cos_mad2 cos_mad3 --single sr_bin cosserver_only --refs cos_bin fltrust --out results/f2_mad
```

| candidate | accuracy | vs `cos_bin` (0.877) | vs `fltrust` (0.873) | wins / ties / losses vs `cosserver_only` |
|---|---:|---:|---:|---|
| `cos_mad2` | 0.840 | −3.73 | −3.36 | 3 / 8 / 8 |
| `cos_mad3` | 0.805 | −7.24 | −6.87 | 3 / 5 / 11 |

## Findings

- **The MAD threshold excludes too few Byzantine clients.** The losses grow with k and are largest
  on `trim_attack` (α = 0.05: −35 and −52 p.p.), `gaussian_noise` (−8 to −11 p.p.) and, for k = 3,
  `sign_flipping` (up to −22 p.p.).
- With 2 Byzantine clients out of 10, the relative-max threshold (b = 0.5) is the better default;
  the MAD threshold is not carried forward.
- The wins (`fltrust_aligned` +14.3 p.p., `low_mag_backdoor` +1 to +3 p.p.) come from excluding
  fewer clients where the server cosine is misleading, not from a better separation.
