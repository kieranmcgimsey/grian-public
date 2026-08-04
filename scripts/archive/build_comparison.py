"""Config-comparison table across every trial and resolution.

Scans one or more trial bases (e.g. the 5-min ``outputs/trials`` and the
30-min ``outputs/trials_30min``), parses each trial name into its
configuration axes — resolution, model family, weather on/off, dispatch mode
— scores it against the matching same-resolution oracle, and writes a single
sortable, self-contained ``outputs/dashboard/comparison.html``.

The point is to make every knob we vary legible side by side: resolution,
forecaster, weather, and how the battery is dispatched (open-loop vs MPC vs
the probabilistic scenario/EV/CVaR family). Capture ratio is always relative
to the perfect-foresight oracle *at that resolution*, so numbers within a
resolution are directly comparable.

Usage::

    python scripts/build_comparison.py
    python scripts/build_comparison.py --out outputs/dashboard/comparison.html
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.sim.analytics import capture_report

# (label, trials-base) pairs to scan. Missing bases are skipped.
# "5min" is the original campaign base (weekly refit, ordinal calendar);
# "5min-v2" is the fresh gold-standard base (monthly refit, one-hot calendar).
BASES = [("5min", "outputs/trials"),
         ("5min-v2", "outputs/trials_5min"),
         ("30min", "outputs/trials_30min")]
REGION = "SA1"

# Human labels for dispatch executors, ordered from simplest to most involved.
DISPATCH_LABEL = {
    "openloop": "open-loop",
    "mpc30": "MPC 30-min",
    "mpc5": "MPC 5-min",
    "mpc_qgate": "qgate (dep.)",
    "mpc_robust": "robust-gate (dep.)",
    "mpc_scenario": "scenario (robust min)",
    "mpc_ev": "scenario EV",
    "mpc_cvar": "scenario CVaR",
}
DISPATCH_ORDER = list(DISPATCH_LABEL)


def _oracle_daily(base: Path) -> pd.Series | None:
    """Load the cached oracle's daily revenue for a base, or None."""
    cache = base / "_oracle" / REGION / "common.parquet"
    if not cache.is_file():
        return None
    sched = pd.read_parquet(cache)
    return sched["revenue"].resample("D").sum()


def _parse(trial: str) -> tuple[str, bool, str]:
    """Split a trial name into (model family, weather flag, executor)."""
    model_part, _, executor = trial.partition("__")
    weather = "_weather" in model_part
    family = model_part.replace("_weather", "")
    return family, weather, executor


def collect() -> pd.DataFrame:
    """Score every trial under every configured base into one table."""
    rows = []
    for res, base_str in BASES:
        base = Path(base_str)
        oracle_daily = _oracle_daily(base)
        if oracle_daily is None:
            continue
        for ledger_path in sorted(base.glob(f"*/{REGION}/ledger.parquet")):
            trial = ledger_path.parent.parent.name
            family, weather, executor = _parse(trial)
            report = capture_report(pd.read_parquet(ledger_path), oracle_daily)
            rows.append({
                "resolution": res,
                "model": family,
                "weather": "✓" if weather else "",
                "dispatch": DISPATCH_LABEL.get(executor, executor),
                "dispatch_key": executor,
                "capture": round(report["capture_ratio"], 4),
                "revenue_m": round(report["total_revenue"] / 1e6, 2),
                "days": len(report["regret_daily"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_d"] = df["dispatch_key"].map(
        lambda k: DISPATCH_ORDER.index(k) if k in DISPATCH_ORDER else 99)
    return df.sort_values(["resolution", "capture"], ascending=[True, False])


def _cap_color(cap: float) -> str:
    """Green-to-grey shade for a capture ratio in [0, 1]."""
    c = max(0.0, min(1.0, cap))
    return f"hsl({int(140 * c)}, {int(45 + 25 * c)}%, {int(28 + 12 * c)}%)"


def render(df: pd.DataFrame, out: str) -> Path:
    """Write the self-contained comparison table HTML."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out_path.write_text("<h1>No trials found.</h1>")
        return out_path

    body = []
    for res in df["resolution"].unique():
        sub = df[df["resolution"] == res]
        best = sub["capture"].max()
        body.append(f"<tr class='sec'><td colspan=6>{res} · "
                    f"{len(sub)} configs · best capture {best:.3f}</td></tr>")
        for _, r in sub.iterrows():
            star = " ★" if r["capture"] == best else ""
            body.append(
                f"<tr><td class='m'>{r['model']}{star}</td>"
                f"<td class='c'>{r['weather']}</td>"
                f"<td>{r['dispatch']}</td>"
                f"<td class='num' style='background:{_cap_color(r['capture'])}'>"
                f"{r['capture']:.3f}</td>"
                f"<td class='num'>${r['revenue_m']:.1f}M</td>"
                f"<td class='num dim'>{r['days']}</td></tr>"
            )
    rows_html = "".join(body)
    css = """
body{margin:0;background:#0f1216;color:#e6e8ec;
     font:14px -apple-system,BlinkMacSystemFont,sans-serif}
header{padding:16px 22px;border-bottom:1px solid #272c34}
h1{font-size:16px;margin:0}
.t{color:#9aa3af;font-size:12px;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px 18px;text-align:left;border-bottom:1px solid #1c2128}
th{color:#9aa3af;font-size:11px;text-transform:uppercase;
   position:sticky;top:0;background:#0f1216}
.sec td{background:#161b22;color:#c7cdd6;font-weight:700;font-size:12px;
        text-transform:uppercase;letter-spacing:.04em}
.m{font-weight:600}
.c{color:#22c55e;text-align:center}
.num{text-align:right;font-family:ui-monospace,Menlo,monospace}
.dim{color:#6b7280}
"""
    caption = ("Capture ratio vs perfect-foresight oracle, per resolution. "
               "★ = best in group. Common window 2025-07 → 2026-06.")
    head = ("<tr><th>Model</th><th>Weather</th><th>Dispatch</th>"
            "<th class='num'>Capture</th><th class='num'>Revenue</th>"
            "<th class='num'>Days</th></tr>")
    out_path.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f"<title>grian · config comparison</title><style>{css}</style>"
        f'</head><body><header><h1>grian · configuration comparison</h1>'
        f'<div class="t">{caption}</div></header>'
        f"<table><thead>{head}</thead><tbody>{rows_html}</tbody></table>"
        f"</body></html>")
    return out_path


def main() -> None:
    """Build the comparison table from all configured bases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/dashboard/comparison.html")
    args = parser.parse_args()
    df = collect()
    path = render(df, args.out)
    if not df.empty:
        print(df.drop(columns=["dispatch_key", "_d"]).to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
