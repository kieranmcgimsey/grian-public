#!/usr/bin/env bash
# LEAR-family baselines, chained AFTER the sprint queue so nothing contends.
# Runs the regularised-linear models (LEAR / ridge / elasticnet / lear_weather)
# at 30-min on the same window/base as everything else, then rebuilds the
# comparison table and landing page so they appear alongside the rest.
#
# Usage: nohup bash scripts/lear_run.sh <wait_pid> > LOG 2>&1 &
set -u
cd "$(dirname "$0")/.."
PID="${1:-}"
RES=30min
MODELS=lear,ridge,elasticnet,lear_weather

ts() { date '+%H:%M:%S'; }
step() { echo "[$(ts)] === $* ==="; }

if [ -n "$PID" ]; then
  step "waiting for prior job (PID $PID) to finish"
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
  step "prior job done"
fi

step "LEAR family: $MODELS @ $RES × {openloop, mpc30}"
python scripts/run_common_eval.py \
  --resolution $RES --refit-days 28 \
  --models $MODELS --executors openloop,mpc30 \
  --base outputs/trials_30min || true

step "rebuild comparison + landing page"
python scripts/build_comparison.py || true
python scripts/build_home.py || true

step "mark board done"
python scripts/jobs.py set learq --status done || true
python scripts/jobs.py render || true

step "LEAR RUN COMPLETE"
