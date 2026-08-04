#!/usr/bin/env bash
# Recompute the autoregression baseline under all calendar encodings.
# The baseline now defaults to one-hot (was ordinal), so the stale ordinal
# trials are removed and re-run, plus the Fourier and ordinal variants for a
# like-for-like comparison. Chained last.
#
# Usage: nohup bash scripts/ar_encoding_run.sh <wait_pid> > LOG 2>&1 &
set -u
cd "$(dirname "$0")/.."
PID="${1:-}"
RES=30min

ts() { date '+%H:%M:%S'; }
step() { echo "[$(ts)] === $* ==="; }

if [ -n "$PID" ]; then
  step "waiting for prior job (PID $PID) to finish"
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
  step "prior job done"
fi

step "drop stale ordinal autoregression trials (baseline is now one-hot)"
rm -rf outputs/trials_30min/autoregression__openloop \
       outputs/trials_30min/autoregression__mpc30

step "autoregression {one-hot, fourier, ordinal} @ $RES × {openloop, mpc30}"
python scripts/run_common_eval.py \
  --resolution $RES --refit-days 28 \
  --models autoregression,autoregression_fourier,autoregression_ordinal \
  --executors openloop,mpc30 --base outputs/trials_30min || true

step "rebuild comparison + landing page"
python scripts/build_comparison.py || true
python scripts/build_home.py || true

step "mark board done"
python scripts/jobs.py set arenc --status done || true
python scripts/jobs.py render || true

step "AR ENCODING RUN COMPLETE"
