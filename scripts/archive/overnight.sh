#!/usr/bin/env bash
# Single-dashboard overnight orchestrator. Everything folds into ONE rich
# dashboard: outputs/dashboard/index.html (build_dashboard on the 30-min base).
#
# Phase 1: when the fast streams (A1 LEAR, A2 AR, C GBM-grid) finish, rebuild
#          index.html so there is a clean thing to review before bed.
# Phase 2: fill cores with the 5-min gold-standard run (data for later).
# Phase 3: when B (linear-quantile fans) finishes, re-grid ALL fan models over
#          the cached fans — the new grid writes native trial artifacts, so the
#          probabilistic executors (scenario/EV/CVaR/mean-CVaR) show up IN the
#          one dashboard with plots/filter/hover. Then rebuild index.html.
#
# Usage: nohup bash scripts/overnight.sh <A1_pid> <A2_pid> <C_pid> <B_pid> > LOG 2>&1 &
set -u
cd "$(dirname "$0")/.."
A1="${1}"; A2="${2}"; C="${3}"; B="${4}"
FANMODELS=lightgbm_qmean,lightgbm_qmean_weather,lear_qmean,lear_qmean_weather,lear_qmean_fourier
SC="${TMPDIR:-/tmp}/grian-scratch"
ts() { date '+%H:%M:%S'; }
step() { echo "[$(ts)] === $* ==="; }
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
rebuild() { python scripts/build_dashboard.py --base outputs/trials_30min \
            --out outputs/dashboard/index.html || true; }

step "Phase 1: waiting for fast streams A1=$A1 A2=$A2 C=$C"
wait_pid "$A1"; wait_pid "$A2"; wait_pid "$C"
step "fast streams done — rebuild index.html (bedtime review package)"
rebuild
python scripts/jobs.py set reviewpkg --status done || true
python scripts/jobs.py render || true
step "REVIEW PACKAGE READY — open outputs/dashboard/index.html"

step "Phase 2: overnight 5-min gold-standard run (data for later, fresh base)"
python scripts/run_common_eval.py --resolution 5min --refit-days 28 \
  --models naive_similar_day --executors openloop --base outputs/trials_5min \
  > "$SC/p5_oracle.log" 2>&1 || true   # pre-cache the slow 5-min oracle
python scripts/run_common_eval.py --resolution 5min --refit-days 28 \
  --models autoregression,lear,lear_weather --executors openloop,mpc30 \
  --base outputs/trials_5min > "$SC/p5_a.log" 2>&1 &
S1=$!
python scripts/run_common_eval.py --resolution 5min --refit-days 28 \
  --models lightgbm_rich,lightgbm_rich_weather --executors openloop,mpc30 \
  --base outputs/trials_5min > "$SC/p5_b.log" 2>&1 &
S2=$!
python scripts/run_common_eval.py --resolution 5min --refit-days 28 \
  --models lightgbm_qmean,lightgbm_qmean_weather --executors openloop,mpc30 \
  --base outputs/trials_5min > "$SC/p5_c.log" 2>&1 &
S3=$!

step "Phase 3: waiting for B=$B (linear-quantile fans) to finish"
wait_pid "$B"
step "B done — re-grid all fan models over cached fans → native executor trials"
python scripts/testbed.py grid --resolution 30min --models "$FANMODELS" \
  --windows full > "$SC/regrid_executors.log" 2>&1 || true
step "rebuild index.html WITH probabilistic executors folded in"
rebuild

step "waiting for 5-min gold streams S1=$S1 S2=$S2 S3=$S3"
wait_pid "$S1"; wait_pid "$S2"; wait_pid "$S3"
step "final rebuild"
rebuild
python scripts/jobs.py set overnightq --status done || true
python scripts/jobs.py render || true
step "OVERNIGHT COMPLETE — single dashboard: outputs/dashboard/index.html"
