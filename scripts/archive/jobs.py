"""Ultra-light local job board: a JSON registry + a self-refreshing HTML view.

No server, no wandb — just ``outputs/jobs/jobs.json`` (the source of truth) and
``outputs/jobs/index.html`` (open it in a browser; it auto-reloads). A ``watch``
loop keeps the HTML fresh by tailing each job's log for its latest progress line.

Commands::

    python scripts/jobs.py add ID --title T --purpose P --log PATH [--pattern RE]
    python scripts/jobs.py set ID --status running|done|failed|queued
    python scripts/jobs.py rm ID
    python scripts/jobs.py list
    python scripts/jobs.py render            # write index.html once
    python scripts/jobs.py watch [--every 5] # keep index.html live
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

DIR = Path("outputs/jobs")
REG = DIR / "jobs.json"
HTML = DIR / "index.html"
COLORS = {"running": "#3b82f6", "done": "#22c55e",
          "failed": "#ef4444", "queued": "#9ca3af"}


def _load() -> list[dict]:
    return json.loads(REG.read_text()) if REG.exists() else []


def _save(jobs: list[dict]) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(jobs, indent=2))


def _fmt_eta(seconds: float) -> str:
    """Human-readable duration, e.g. '3h 20m' or '45m'."""
    if seconds < 0:
        return "—"
    h, m = int(seconds // 3600), int((seconds % 3600) // 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m" if m else "<1m"


def _matrix_progress(job: dict) -> dict | None:
    """Weighted trial progress + ETA for a run declaring a ``matrix``.

    The matrix is ``{"base": dir, "region": R, "weights": {trial: weight}}``
    where each ``weight`` is the trial's relative cost (openloop 1, mpc30
    ~2.3, scenario ~9 — so the ETA reflects that the slow scenario trials
    run last). A trial counts as done when its ``metrics.json`` exists.

    The ETA rate comes from the runner's own ``"<trial> ran in <secs>s"``
    log lines — the actual measured cost of the trials this run computed
    (skipped pre-existing trials emit no such line, so they don't distort
    the rate). Cost is divided by the trial's weight to get seconds per unit
    weight, then multiplied by the remaining weight.

    Returns:
        Dict with n_done, n_total, pct (0-100), and eta (str), or None if
        the job declares no matrix.
    """
    matrix = job.get("matrix")
    if not matrix:
        return None
    base = Path(matrix["base"])
    region = matrix.get("region", "SA1")
    weights = matrix["weights"]

    total_w = sum(weights.values())
    done_w = 0.0
    n_done = 0
    for name, w in weights.items():
        if (base / name / region / "metrics.json").is_file():
            n_done += 1
            done_w += w

    # Measured seconds-per-weight from this run's "<trial> ran in <s>s" lines.
    ran, weighed = 0.0, 0.0
    log = Path(job.get("log", ""))
    if log.is_file():
        for m in re.finditer(r"(\S+) ran in ([\d.]+)s", _tail_text(log)):
            if m.group(1) in weights:
                ran += float(m.group(2))
                weighed += weights[m.group(1)]

    pct = 100.0 * done_w / total_w if total_w else 0.0
    if done_w >= total_w:
        eta = "done"
    elif weighed > 0:
        sec_per_weight = ran / weighed
        eta = "~" + _fmt_eta((total_w - done_w) * sec_per_weight)
    else:
        eta = "—"
    return {"n_done": n_done, "n_total": len(weights),
            "pct": round(pct), "eta": eta}


def _tail_text(log: Path, max_bytes: int = 262144) -> str:
    """Read only the last ``max_bytes`` of a (possibly huge) log file.

    Run logs fill with library warning-spam and reach many MB; reading the
    whole file on every render is slow. The progress signals we want (latest
    matching line, recent "ran in" times) live at the tail, so a bounded read
    keeps the board fast.
    """
    size = log.stat().st_size
    with open(log, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode(errors="ignore")


def _progress(job: dict) -> str:
    """Latest progress line from the job's log (matching its pattern, if any)."""
    log_path = job.get("log", "")
    if not log_path:
        return ""
    log = Path(log_path)
    if not log.is_file():
        return ""
    lines = _tail_text(log).splitlines()
    pat = job.get("pattern")
    if pat:
        for ln in reversed(lines):
            if re.search(pat, ln):
                return ln.strip()[-140:]
    return lines[-1].strip()[-140:] if lines else ""


def render() -> None:
    """Write the self-contained, auto-reloading HTML board."""
    jobs = _load()
    rows = []
    for j in jobs:
        st = j.get("status", "queued")
        dot = COLORS.get(st, "#9ca3af")
        prog = (_progress(j) or "—").replace("<", "&lt;")
        mp = _matrix_progress(j)
        if mp:
            bar = (
                f"<div class='bar'><span style='width:{mp['pct']}%;"
                f"background:{dot}'></span></div>"
                f"<div class='meta'>{mp['n_done']}/{mp['n_total']} trials · "
                f"{mp['pct']}% · ETA {mp['eta']}</div>"
            )
            prog_cell = f"<td class='g'>{bar}<div class='ln'>{prog}</div></td>"
        else:
            prog_cell = f"<td class='g'>{prog}</td>"
        rows.append(
            f"<tr><td class='j'><b>{j['title']}</b>"
            f"<div class='p'>{j.get('purpose', '')}</div></td>"
            f"<td><span class='pill' style='background:{dot}'>{st}</span></td>"
            f"{prog_cell}</tr>"
        )
    body = "".join(rows) or "<tr><td colspan=3 class='p'>No jobs.</td></tr>"
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>grian jobs</title><style>
body{{margin:0;background:#0f1216;color:#e6e8ec;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{padding:14px 20px;border-bottom:1px solid #272c34;display:flex;gap:12px;align-items:baseline}}
h1{{font-size:16px;margin:0}} .t{{color:#9aa3af;font-size:12px}}
table{{width:100%;border-collapse:collapse}} td{{padding:11px 20px;border-bottom:1px solid #1c2128;vertical-align:top}}
.j b{{font-size:14px}} .p{{color:#9aa3af;font-size:12px;margin-top:3px;max-width:520px}}
.pill{{color:#0f1216;font-weight:700;font-size:11px;padding:2px 9px;border-radius:20px;text-transform:uppercase}}
.g{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#c7cdd6}}
.bar{{height:7px;background:#1c2128;border-radius:4px;overflow:hidden;max-width:320px}}
.bar span{{display:block;height:100%;border-radius:4px}}
.meta{{font-size:11px;color:#9aa3af;margin-top:4px}}
.ln{{margin-top:5px;color:#6b7280;font-size:11px;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
</style></head><body>
<header><h1>grian · job board</h1><span class="t">{len(jobs)} jobs · auto-refresh 5s · updated {datetime.now():%H:%M:%S}</span></header>
<table><tbody>{body}</tbody></table></body></html>""")


def main() -> None:
    """Dispatch a board command."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("id")
    a.add_argument("--title", required=True)
    a.add_argument("--purpose", default="")
    a.add_argument("--log", default="")
    a.add_argument("--pattern", default="")
    a.add_argument("--status", default="running")
    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("--status", required=True)
    r = sub.add_parser("rm")
    r.add_argument("id")
    sub.add_parser("list")
    sub.add_parser("render")
    w = sub.add_parser("watch")
    w.add_argument("--every", type=int, default=5)
    args = p.parse_args()

    jobs = _load()
    if args.cmd == "add":
        jobs = [j for j in jobs if j["id"] != args.id]
        jobs.append({"id": args.id, "title": args.title, "purpose": args.purpose,
                     "log": args.log, "pattern": args.pattern, "status": args.status,
                     "added": datetime.now().strftime("%Y-%m-%d %H:%M")})
        _save(jobs)
        render()
    elif args.cmd == "set":
        for j in jobs:
            if j["id"] == args.id:
                j["status"] = args.status
        _save(jobs)
        render()
    elif args.cmd == "rm":
        _save([j for j in jobs if j["id"] != args.id])
        render()
    elif args.cmd == "list":
        for j in jobs:
            print(f"{j['id']:24} {j.get('status',''):8} {j['title']}")
    elif args.cmd == "render":
        render()
    elif args.cmd == "watch":
        while True:
            render()
            time.sleep(args.every)


if __name__ == "__main__":
    main()
