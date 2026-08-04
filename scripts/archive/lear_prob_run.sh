#!/usr/bin/env bash
# Probabilistic LEAR: linear quantile fan → the same dispatch grid as the GBM
# quantile model, via the fan-cache test bed. Chained after the point LEAR run.
#
# Usage: nohup bash scripts/lear_prob_run.sh <wait_pid> > LOG 2>&1 &
set -u
cd "$(dirname "$0")/.."
PID="${1:-}"
RES=30min
MODELS=lear_qmean,lear_qmean_weather
WINDOWS=full,spike_jan26,calm_sep25

ts() { date '+%H:%M:%S'; }
step() { echo "[$(ts)] === $* ==="; }

if [ -n "$PID" ]; then
  step "waiting for prior job (PID $PID) to finish"
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
  step "prior job done"
fi

step "testbed BUILD linear quantile fans ($MODELS @ $RES)"
python scripts/testbed.py build --resolution $RES --models $MODELS --windows $WINDOWS || true

step "testbed GRID (dispatch ladder + CVaR lambda sweep, linear fan)"
python scripts/testbed.py grid --resolution $RES --models $MODELS --windows $WINDOWS || true

step "forecast quality for the linear fans"
python scripts/forecast_quality.py --resolution $RES || true

step "rebuild test-bed + comparison + landing dashboards"
python scripts/build_testbed_dashboard.py --resolution $RES || true
python scripts/build_comparison.py || true
python scripts/build_home.py || true

step "mark board done"
python scripts/jobs.py set learprob --status done || true
python scripts/jobs.py render || true

step "PROB-LEAR RUN COMPLETE"
