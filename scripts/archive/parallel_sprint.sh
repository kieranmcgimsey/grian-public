#!/usr/bin/env bash
# Maximise CPU: run the remaining work as parallel streams instead of a chain.
# Streams write disjoint outputs (per-trial dirs / per-model fan files /
# per-invocation results), so they never collide. Cached fans mean the
# executor sweeps are pure LP replays.
#
#   A1  LEAR point family        (run_common_eval → trials_30min)   [heavy: LassoCV]
#   A2  autoregression encodings (run_common_eval → trials_30min)   [fast]
#   B   linear-quantile fans     (testbed build+grid)               [heavy: QuantileRegressor]
#   C   GBM executor sweep       (testbed grid over cached fans, incl mean-CVaR) [LP-light]
#
# Usage: nohup bash scripts/parallel_sprint.sh > LOG 2>&1 &
set -u
cd "$(dirname "$0")/.."
RES=30min
SC="${TMPDIR:-/tmp}/grian-scratch"
WINDOWS=full,spike_jan26,calm_sep25
ts() { date '+%H:%M:%S'; }
step() { echo "[$(ts)] === $* ==="; }

step "drop stale ordinal autoregression trials (baseline is now one-hot)"
rm -rf outputs/trials_30min/autoregression__openloop \
       outputs/trials_30min/autoregression__mpc30

step "pre-compute test-bed window oracles (avoid a compute race between grids)"
python - <<'PY' || true
import sys; sys.path.insert(0, "scripts")
import pandas as pd
import testbed
testbed.RESOLUTION = "30min"
testbed.DATA_PATH = "data/processed/SA1_30min_sim.parquet"
d = pd.read_parquet(testbed.DATA_PATH)
for w in ("full", "spike_jan26", "calm_sep25"):
    testbed._window_oracle(d, w)
    print("oracle ready:", w)
PY

step "launch 4 parallel streams"
python scripts/run_common_eval.py --resolution $RES --refit-days 28 \
  --models lear,ridge,elasticnet,lear_weather,lear_fourier,lear_ordinal \
  --executors openloop,mpc30 --base outputs/trials_30min > "$SC/p_lear.log" 2>&1 &
A1=$!
python scripts/run_common_eval.py --resolution $RES --refit-days 28 \
  --models autoregression,autoregression_fourier,autoregression_ordinal \
  --executors openloop,mpc30 --base outputs/trials_30min > "$SC/p_ar.log" 2>&1 &
A2=$!
( python scripts/testbed.py build --resolution $RES \
    --models lear_qmean,lear_qmean_weather,lear_qmean_fourier --windows $WINDOWS \
  && python scripts/testbed.py grid --resolution $RES \
    --models lear_qmean,lear_qmean_weather,lear_qmean_fourier --windows $WINDOWS \
) > "$SC/p_learq.log" 2>&1 &
B=$!
python scripts/testbed.py grid --resolution $RES \
  --models lightgbm_qmean,lightgbm_qmean_weather --windows $WINDOWS \
  > "$SC/p_gbmsweep.log" 2>&1 &
C=$!

step "streams: A1(LEAR)=$A1 A2(AR)=$A2 B(learQ)=$B C(GBMsweep)=$C — waiting"
wait $A1 $A2 $B $C
step "all streams done"

step "rebuild all dashboards"
python scripts/forecast_quality.py --resolution $RES || true
python scripts/build_testbed_dashboard.py --resolution $RES || true
python scripts/build_comparison.py || true
python scripts/build_home.py || true
python scripts/jobs.py set parallelq --status done || true
python scripts/jobs.py render || true
step "PARALLEL SPRINT COMPLETE"
