#!/usr/bin/env bash
# Sprint experiment queue: runs sequentially AFTER the point matrix finishes,
# so nothing contends for cores. Each step is best-effort (|| true) so one
# failure does not abort the queue. Progress is echoed with timestamps.
#
# Usage: nohup bash scripts/sprint_queue.sh <point_matrix_pid> > LOG 2>&1 &
set -u
cd "$(dirname "$0")/.."
PID="${1:-}"
RES=30min
MODELS=lightgbm_qmean,lightgbm_qmean_weather
WINDOWS=full,spike_jan26,calm_sep25

ts() { date '+%H:%M:%S'; }
step() { echo "[$(ts)] === $* ==="; }

if [ -n "$PID" ]; then
  step "waiting for point matrix (PID $PID) to finish"
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
  step "point matrix done"
fi

step "testbed BUILD fans ($MODELS @ $RES, windows=$WINDOWS)"
python scripts/testbed.py build --resolution $RES --models $MODELS --windows $WINDOWS || true

step "testbed GRID (dispatch ladder + CVaR lambda sweep)"
python scripts/testbed.py grid --resolution $RES --models $MODELS --windows $WINDOWS || true

step "forecast quality (CRPS + coverage) from cached fans"
python scripts/forecast_quality.py --resolution $RES || true

step "render test-bed dashboard (dispatch ladder + lambda sweep + fan quality)"
python scripts/build_testbed_dashboard.py --resolution $RES || true

step "rebuild config comparison table"
python scripts/build_comparison.py || true

step "rebuild main dashboard (30-min trials)"
python scripts/build_dashboard.py --base outputs/trials_30min \
  --out outputs/dashboard/index_30min.html || true

step "rebuild landing page"
python scripts/build_home.py || true

step "mark board jobs done"
python scripts/jobs.py set eval30 --status done || true
python scripts/jobs.py set sprintq --status done || true
python scripts/jobs.py render || true

step "SPRINT QUEUE COMPLETE"
