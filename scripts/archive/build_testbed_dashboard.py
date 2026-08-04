"""Test-bed dashboard subpage: dispatch ladder, CVaR lambda sweep, fan quality.

Reads the test bed's replay results (``results_<res>.json``) and forecast
quality (``forecast_quality_<res>.json``) and renders a single self-contained
``outputs/dashboard/testbed.html``: for each (model, window) it shows the
capture of every dispatch variant as a bar, highlights the CVaR lambda sweep
from risk-neutral (EV) to worst-case (robust), and lists the fan's CRPS and
coverage. Safe to run before the data lands — it renders a "pending" note.

Usage::

    python scripts/build_testbed_dashboard.py --resolution 30min
"""

import argparse
import json
from pathlib import Path

OUT = Path("outputs/testbed")
DASH = Path("outputs/dashboard/testbed.html")

# Display order + labels for dispatch variants.
ORDER = ["point", "robust", "ev",
         "cvar_010", "cvar_025", "cvar_050", "cvar_075", "cvar_090",
         "meancvar_l50_a50", "meancvar_l50_a10", "meancvar_l90_a10"]
LABEL = {"point": "point MPC", "robust": "robust (min)", "ev": "EV (blend)",
         "cvar_010": "CVaR-blend λ.10", "cvar_025": "CVaR-blend λ.25",
         "cvar_050": "CVaR-blend λ.50", "cvar_075": "CVaR-blend λ.75",
         "cvar_090": "CVaR-blend λ.90",
         "meancvar_l50_a50": "mean-CVaR LP λ.5/α.5",
         "meancvar_l50_a10": "mean-CVaR LP λ.5/α.1",
         "meancvar_l90_a10": "mean-CVaR LP λ.9/α.1"}


def _load(name: str) -> object:
    path = OUT / name
    return json.loads(path.read_text()) if path.is_file() else None


def _load_results(res: str) -> list | None:
    """Merge every per-invocation results file for a resolution (+ legacy)."""
    rows: list = []
    legacy = OUT / f"results_{res}.json"
    if legacy.is_file():
        rows += json.loads(legacy.read_text())
    for path in sorted((OUT / "results").glob(f"results_{res}__*.json")):
        rows += json.loads(path.read_text())
    # De-dup on (model, window, dispatch); later files win.
    dedup = {(r["model"], r["window"], r["dispatch"]): r for r in rows}
    return list(dedup.values()) or None


def _bar(cap: float, best: float) -> str:
    """A capture bar scaled to the group best."""
    pct = 100 * cap / best if best > 0 else 0
    hue = int(140 * max(0.0, min(1.0, cap)))
    return (f"<div class='bar'><span style='width:{pct:.0f}%;"
            f"background:hsl({hue},55%,42%)'></span></div>")


def render(res: str) -> Path:
    """Build the test-bed HTML for one resolution."""
    results = _load_results(res)
    fq = _load(f"forecast_quality_{res}.json") or {}
    DASH.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        DASH.write_text(
            "<h1>grian · test bed</h1><p>No results yet — run "
            "<code>scripts/testbed.py grid</code>. The sprint queue produces "
            "these automatically after the point matrix.</p>")
        return DASH

    # Group rows by (model, window).
    groups: dict[tuple, list] = {}
    for r in results:
        groups.setdefault((r["model"], r["window"]), []).append(r)

    blocks = []
    for (model, window), rows in sorted(groups.items()):
        by = {r["dispatch"]: r for r in rows}
        best = max(r["capture"] for r in rows)
        ordered = [d for d in ORDER if d in by] + [
            d for d in by if d not in ORDER]
        trs = []
        for d in ordered:
            r = by[d]
            star = " ★" if r["capture"] == best else ""
            trs.append(
                f"<tr><td>{LABEL.get(d, d)}{star}</td>"
                f"<td class='num'>{r['capture']:.3f}</td>"
                f"<td>{_bar(r['capture'], best)}</td>"
                f"<td class='num dim'>${r['revenue'] / 1e6:.1f}M</td></tr>")
        key = f"{model}__{window}"
        fqi = fq.get(key)
        fq_html = ""
        if fqi:
            cov = " ".join(f"{q}:{c:.2f}" for q, c in fqi["coverage"].items())
            fq_html = (f"<div class='fq'>fan quality — CRPS ${fqi['crps']:.1f}"
                       f" · coverage {cov}</div>")
        blocks.append(
            f"<section><h2>{model} · {window}</h2>"
            f"<table><tbody>{''.join(trs)}</tbody></table>{fq_html}</section>")

    DASH.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>grian · test bed ({res})</title><style>"
        "body{margin:0;background:#0f1216;color:#e6e8ec;"
        "font:14px -apple-system,BlinkMacSystemFont,sans-serif}"
        "header{padding:16px 22px;border-bottom:1px solid #272c34}"
        "h1{font-size:16px;margin:0} h2{font-size:13px;color:#c7cdd6;margin:0 0 8px}"
        ".t{color:#9aa3af;font-size:12px;margin-top:4px}"
        "section{padding:16px 22px;border-bottom:1px solid #1c2128}"
        "table{width:100%;max-width:640px;border-collapse:collapse}"
        "td{padding:5px 10px;border-bottom:1px solid #171c22;font-size:13px}"
        ".num{text-align:right;font-family:ui-monospace,Menlo,monospace}"
        ".dim{color:#6b7280}"
        ".bar{height:8px;background:#1c2128;border-radius:4px;overflow:hidden;width:200px}"
        ".bar span{display:block;height:100%}"
        ".fq{color:#9aa3af;font-size:12px;margin-top:8px;font-family:ui-monospace,Menlo,monospace}"
        "</style></head><body>"
        f"<header><h1>grian · dispatch test bed ({res})</h1>"
        "<div class='t'>Capture by dispatch variant over a frozen forecast fan. "
        "CVaR λ sweeps risk-neutral (EV) → worst-case (robust). "
        "★ = best per group.</div>"
        f"</header>{''.join(blocks)}</body></html>")
    return DASH


def main() -> None:
    """Render the test-bed dashboard for the given resolution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="30min", choices=["5min", "30min"])
    args = parser.parse_args()
    path = render(args.resolution)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
