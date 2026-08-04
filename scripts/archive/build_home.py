"""Landing page tying the grian dashboards together for a demo.

Writes a self-contained ``outputs/dashboard/home.html`` linking the config
comparison, the dispatch test bed, the interactive trading dashboard, and the
live job board, with a one-line description of each and the pipeline story.
Purely static links — safe to (re)run any time.

Usage::

    python scripts/build_home.py
"""

from pathlib import Path

DASH = Path("outputs/dashboard/home.html")

CARDS = [
    ("Configuration comparison", "comparison.html",
     "Capture ratio for every model × weather × dispatch config, per "
     "resolution — the headline table."),
    ("Dispatch test bed", "testbed.html",
     "Probabilistic dispatch ladder (robust / EV / CVaR-λ) over a frozen "
     "forecast fan, plus fan CRPS & coverage."),
    ("Trading dashboard (30-min)", "index_30min.html",
     "Interactive ledger: forecast vs actual price, SOC, and daily revenue "
     "over the test window."),
    ("Job board", "../jobs/index.html",
     "Live run status with weighted progress bars and ETAs."),
]

PIPELINE = [
    "Raw 5-min NEM SA1 prices → mean-resampled to 30-min settlement prices.",
    "Forecast: naive / autoregression / LightGBM (rich features ± weather) / "
    "LightGBM quantile fan.",
    "Dispatch a 100 MW / 2 h battery: open-loop, receding-horizon MPC, and "
    "probabilistic (scenario / EV / CVaR).",
    "Score against a perfect-foresight oracle → capture ratio; score the fan "
    "with CRPS & coverage.",
]


def main() -> None:
    """Write the landing page."""
    DASH.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(
        f'<a class="card" href="{href}"><h2>{title}</h2><p>{desc}</p></a>'
        for title, href, desc in CARDS)
    steps = "".join(f"<li>{s}</li>" for s in PIPELINE)
    DASH.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>grian · NEM price forecasting & battery dispatch</title><style>"
        "body{margin:0;background:#0f1216;color:#e6e8ec;"
        "font:15px -apple-system,BlinkMacSystemFont,sans-serif;line-height:1.5}"
        "header{padding:28px 26px;border-bottom:1px solid #272c34}"
        "h1{font-size:20px;margin:0} .t{color:#9aa3af;font-size:13px;margin-top:6px}"
        "main{padding:22px 26px;max-width:900px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));"
        "gap:14px;margin-bottom:26px}"
        ".card{display:block;padding:16px;background:#161b22;border:1px solid #272c34;"
        "border-radius:10px;text-decoration:none;color:inherit}"
        ".card:hover{border-color:#3b82f6}"
        ".card h2{font-size:14px;margin:0 0 6px;color:#e6e8ec}"
        ".card p{margin:0;color:#9aa3af;font-size:12.5px}"
        "h3{font-size:13px;color:#c7cdd6;text-transform:uppercase;letter-spacing:.04em}"
        "ol{color:#c7cdd6;font-size:13.5px;padding-left:20px}"
        "ol li{margin-bottom:6px}"
        "</style></head><body>"
        "<header><h1>grian · NEM electricity price forecasting → battery dispatch</h1>"
        "<div class='t'>SA1 · forecast-driven battery arbitrage · capture ratio vs "
        "perfect-foresight oracle</div></header>"
        f"<main><div class='grid'>{cards}</div>"
        f"<h3>Pipeline</h3><ol>{steps}</ol></main></body></html>")
    print(f"Wrote {DASH}")


if __name__ == "__main__":
    main()
