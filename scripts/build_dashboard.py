"""Build the static HTML trading dashboard from trial artifacts.

Reads every trial under the ``--base`` directory (default the current 30-min
grid, ``outputs/trials_30min/``) and writes a single self-contained
``outputs/dashboard/index.html`` — open it directly in a browser (no server).
Rebuild whenever new trials land.

Usage::

    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --base outputs/trials --region SA1  # legacy 5-min
"""

import argparse

from grian.sim.dashboard import DEFAULT_BUFFER_DAYS, build_dashboard


def main() -> None:
    """Parse arguments and build the dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to the current 30-min grid — the legacy 5-min dir is outputs/trials.
    parser.add_argument("--base", default="outputs/trials_30min",
                        help="Trials directory")
    parser.add_argument(
        "--out", default="outputs/dashboard/index.html", help="Output HTML",
    )
    parser.add_argument("--region", default="SA1", help="Restrict to one NEM region")
    parser.add_argument(
        "--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS,
        help="Days of 5-minute recent detail to embed",
    )
    args = parser.parse_args()
    path = build_dashboard(
        base=args.base, out=args.out,
        region_filter=args.region, buffer_days=args.buffer_days,
    )
    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
