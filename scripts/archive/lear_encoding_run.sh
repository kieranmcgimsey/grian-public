#!/usr/bin/env bash
# Calendar-encoding ablation for LEAR: ordinal vs one-hot vs Fourier.
# "lear" (one-hot) already runs in the point + prob LEAR jobs; this adds the
# ordinal (before) and Fourier variants so the comparison table shows all three
# side by side, for both point and probabilistic LEAR. Chained last.
#
# Usage: nohup bash scripts/lear_encoding_run.sh <wait_pid> > LOG 2>&1 &
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

step "point LEAR encodings: lear_fourier, lear_ordinal @ $RES × {openloop, mpc30}"
python scripts/run_common_eval.py \
  --resolution $RES --refit-days 28 \
  --models lear_fourier,lear_ordinal --executors openloop,mpc30 \
  --base outputs/trials_30min || true

step "probabilistic LEAR (Fourier) fan build + dispatch grid"
python scripts/testbed.py build --resolution $RES \
  --models lear_qmean_fourier --windows full,spike_jan26,calm_sep25 || true
python scripts/testbed.py grid --resolution $RES \
  --models lear_qmean_fourier --windows full,spike_jan26,calm_sep25 || true

step "rebuild dashboards"
python scripts/forecast_quality.py --resolution $RES || true
python scripts/build_testbed_dashboard.py --resolution $RES || true
python scripts/build_comparison.py || true
python scripts/build_home.py || true

step "mark board done"
python scripts/jobs.py set learenc --status done || true
python scripts/jobs.py render || true

step "LEAR ENCODING ABLATION COMPLETE"
