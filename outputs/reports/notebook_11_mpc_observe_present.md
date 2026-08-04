# Notebook 11: Making the MPC work (observe_present bug + fix)

Window: 30-min SA1 common window (2025-07 to 2026-06) and regime sub-windows.

## Headline (full window, capture ratio)
- naive 30-min MPC (the bug): 0.259
- open-loop (commit day-ahead): 0.537
- AR point bar: 0.543
- spike-gated MPC (gate $3000): 0.569

## Root cause
`observe_present` pinned step 0 of every re-solve to the true price while the
rest of the horizon stayed forecast, so the LP traded the forecast residual
(mean-reversion noise). Confirmed by: (a) damage tracks resolve, not reforecast;
(b) survives removing the throughput budget; (c) observe_present=OFF makes
resolve=1 reproduce open-loop exactly (Bellman).

## Fix and validation
Spike-gated observe (`observe_gate`): pin the true price only for prices above
the gate. Cross-validated (beats-or-ties open-loop on every regime):

| window | open-loop | spike-gated MPC | delta |
|---|---|---|---|
| spike_jan26 | 0.576 | 0.656 | +0.080 |
| calm_sep25 | 0.648 | 0.652 | +0.004 |
| full | 0.537 | 0.569 | +0.032 |

See experiment_log.md Entries 034-035 for the full record.