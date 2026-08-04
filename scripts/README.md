# scripts/ — the current entry points

Six scripts, in the order you'd run them. Everything else lives in
[`archive/`](archive/) (campaign-era one-offs, kept for reference, unmaintained).

| Script | What it does |
|---|---|
| [`build_sim_data.py`](build_sim_data.py) | Build the processed price/demand parquet the sim reads (`data/processed/SA1_*_sim.parquet`). |
| [`build_weather_data.py`](build_weather_data.py) | Add ERA5 weather features (needs a Copernicus CDS key; see the top-level README). |
| [`run_common_eval.py`](run_common_eval.py) | Evaluate **point** models over the shared held-out year (open-loop / mpc30 / spike-gated MPC). Writes per-trial ledgers to `outputs/trials_30min/`. |
| [`testbed.py`](testbed.py) | Build **quantile fans** (`testbed.py build`) and replay dispatch executors over them (`testbed.py grid`). `--fan-cadence` and `--executors` control cost. |
| [`balanced_eval.py`](balanced_eval.py) | Score trials by **balanced** (per-month) and **pooled** capture, with a per-month heatmap. |
| [`build_dashboard.py`](build_dashboard.py) | Assemble the self-contained interactive dashboard from the trial artifacts → `outputs/dashboard/index.html`. |

## Typical flow

```bash
# 1. data (once)
python scripts/build_sim_data.py
python scripts/build_weather_data.py            # optional (weather features)

# 2. evaluate point models on the common window
python scripts/run_common_eval.py --resolution 30min --base outputs/trials_30min \
  --executors openloop,mpc30,mpc_spike

# 3. build + score quantile fans (daily cadence, just the ablation executors)
python scripts/testbed.py build --models lightgbm_qmean_weather_fourier,lear_qmean_torch_fourier \
  --windows full --resolution 30min --fan-cadence 48
python scripts/testbed.py grid  --models <same> --windows full --resolution 30min \
  --fan-cadence 48 --executors point_ol,mpc_spikegate

# 4. metric + dashboard
python scripts/balanced_eval.py --base outputs/trials_30min --trials <trial names> --out my_eval
python scripts/build_dashboard.py --base outputs/trials_30min --region SA1
```

See [`docs/running-experiments.md`](../docs/running-experiments.md) for detail, and
[`docs/README.md`](../docs/README.md) for the mental model.

## archive/

Campaign-era scripts (individual baselines, MPC-ablation runners, old dashboard
builders, sprint shell scripts, one-off report/plot utilities). Superseded by the
six above; kept for provenance. Not maintained — run at your own risk.
