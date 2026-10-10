# Figures

Regenerate with:

```bash
python3 figures/make_figures.py
```

**No number in any figure is typed in by hand.** `make_figures.py` reads the frozen aggregates
under `results/` and asserts the expected values, so a mismatch fails loudly rather than
producing a silently wrong chart.

| Figure | Source artifact | What it shows |
|---|---|---|
| `fig1_exp08_throughput_vs_mbt.png` | `results/exp08_condition_summary.csv` | output throughput vs MBT, one line per concurrency level (C1/C2/C4/C8) |
| `fig2_exp11_throughput_advantage_vs_reuse.png` | `results/exp11_interaction_summary.csv` (`metric=out_tok_s`, 512→4096) | the larger-MBT throughput advantage shrinking as reuse grows; PHIGH marked *not block-consistent* |
| `fig3_exp11_ttft_effect_vs_reuse.png` | `results/exp11_interaction_summary.csv` (`metric=ttft_ms`, 512→4096) | the TTFT effect flipping from a small benefit into a penalty |
| `fig4_exp12_prefill_transition.png` | `results/exp12_hypothesis_status.csv` | `prepare_prefill` marker count 9 → 5, and steady decode spacing essentially unchanged (34.200 vs 34.412 ms) |

## Reading note for `fig2`

Bars are coloured by **block consistency**, not by magnitude: a bar drawn in the "inconsistent"
colour means the three randomized blocks did **not** agree in sign, so the pooled number is not
a robust effect. `PHIGH`'s +0.252% is exactly that case.

## Reading note for `fig4`

The right-hand panel is the **falsification**: if MBT4096 accelerated decode, that spacing would
drop. It does not (34.200 → 34.412 ms). The left-hand panel shows what *did* change — the prefill
structure.
