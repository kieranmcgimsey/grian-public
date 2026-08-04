"""Generate all conceptual figures for the learning guides.

Each function creates one figure and saves it to figures/.
Run: python gen_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

DPI = 150
BLUE = "#2c5f8a"
RED = "#c0392b"
GREEN = "#27ae60"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"
GRAY = "#7f8c8d"
LIGHT_BG = "#f8f9fa"


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  {name}.png")


# ── CHAPTER 01 ──────────────────────────────────────────────

def ch01_interval_timestamps():
    """Show interval-ending vs interval-start convention."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.5), gridspec_kw={"hspace": 0.6})

    for ax_i, (ax, label, arrows_x, color) in enumerate([
        (axes[0], "AEMO convention: interval-ENDING", [0.5, 1.0, 1.5, 2.0], RED),
        (axes[1], "Our convention: interval-START", [0.0, 0.5, 1.0, 1.5], BLUE),
    ]):
        ax.set_xlim(-0.2, 2.7)
        ax.set_ylim(-0.5, 1.2)
        ax.axis("off")
        ax.set_title(label, fontsize=11, fontweight="bold", color=color, loc="left")

        # Draw intervals as boxes
        for i, start in enumerate([0, 0.5, 1.0, 1.5]):
            rect = mpatches.FancyBboxPatch((start, 0), 0.48, 0.6, boxstyle="round,pad=0.02",
                                           facecolor=color, alpha=0.15, edgecolor=color)
            ax.add_patch(rect)
            ax.text(start + 0.24, 0.3, f"Interval {i+1}", ha="center", va="center", fontsize=8)

        # Draw timestamp markers
        for x in arrows_x:
            ax.plot(x, -0.15, "v", color=color, markersize=8)
            t = int(x * 60)
            h, m = divmod(t, 60)
            ax.text(x, -0.4, f"{h:02d}:{m:02d}", ha="center", fontsize=8, color=color)

        # Time axis
        ax.annotate("", xy=(2.5, -0.15), xytext=(-0.1, -0.15),
                     arrowprops=dict(arrowstyle="->", color=GRAY))
        ax.text(2.6, -0.15, "time", fontsize=8, color=GRAY, va="center")

    save(fig, "01_interval_timestamps")


def ch01_arcsinh_transform():
    """Show arcsinh vs log vs identity transforms."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.linspace(-1000, 5000, 1000)

    # Raw price distribution
    np.random.seed(42)
    prices = np.concatenate([
        np.random.lognormal(3.5, 0.8, 5000),
        np.random.uniform(-200, 0, 200),
        np.random.lognormal(7, 1, 50),
    ])
    prices = np.clip(prices, -1000, 17500)

    axes[0].hist(prices, bins=200, color=BLUE, alpha=0.7, edgecolor="none")
    axes[0].set_xlabel("Price ($/MWh)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Raw prices\n(impossible to model directly)", fontsize=10)
    axes[0].set_xlim(-1200, 5000)
    axes[0].axvline(0, color=RED, linewidth=1, linestyle="--", alpha=0.5)

    # Arcsinh transform
    transformed = np.arcsinh(prices)
    axes[1].hist(transformed, bins=100, color=GREEN, alpha=0.7, edgecolor="none")
    axes[1].set_xlabel("arcsinh(price)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("After arcsinh transform\n(much more symmetric)", fontsize=10)

    # The transform function itself
    x_fn = np.linspace(-500, 2000, 500)
    axes[2].plot(x_fn, np.arcsinh(x_fn), color=BLUE, linewidth=2, label="arcsinh(x)")
    axes[2].plot(x_fn, x_fn / 200, color=GRAY, linewidth=1, linestyle="--", label="x/200 (linear)", alpha=0.5)
    axes[2].set_xlabel("Price ($/MWh)", fontsize=10)
    axes[2].set_ylabel("Transformed value", fontsize=10)
    axes[2].set_title("The arcsinh function\n(compresses extremes)", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(-8, 10)
    axes[2].axhline(0, color=GRAY, linewidth=0.5)
    axes[2].axvline(0, color=GRAY, linewidth=0.5)

    fig.tight_layout()
    save(fig, "01_arcsinh_transform")


def ch01_nem_regions():
    """Simplified NEM regions diagram."""
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)

    regions = [
        ("QLD1", 5, 11.5, "#FF9800", "Queensland\nCoal + Gas + Solar"),
        ("NSW1", 6, 8.5, "#2196F3", "New South Wales\nBlack Coal + Solar"),
        ("VIC1", 5, 5.5, "#4CAF50", "Victoria\nBrown Coal + Wind"),
        ("SA1", 2.5, 6.5, "#F44336", "South Australia\nWind + Solar + Gas"),
        ("TAS1", 5.5, 2.5, "#9C27B0", "Tasmania\nHydro (80%+)"),
    ]

    for name, x, y, color, desc in regions:
        rect = mpatches.FancyBboxPatch((x-1.5, y-0.8), 3, 1.8,
                                       boxstyle="round,pad=0.15",
                                       facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.3, name, ha="center", fontsize=14, fontweight="bold", color=color)
        ax.text(x, y-0.3, desc, ha="center", fontsize=8, color="#333")

    # Interconnectors
    for (x1, y1), (x2, y2), label in [
        ((5, 10.5), (6, 9.5), "QNI"),
        ((6, 7.5), (5, 6.5), "VIC-NSW"),
        ((3.5, 5.8), (4, 5.5), "Heywood"),
        ((5, 4.5), (5.5, 3.5), "Basslink"),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="<->", color=GRAY, linewidth=1.5))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx-0.6, my, label, fontsize=7, color=GRAY, style="italic")

    ax.set_title("The National Electricity Market (NEM)\nFive interconnected pricing regions",
                 fontsize=13, fontweight="bold", pad=10)
    save(fig, "01_nem_regions")


def ch01_price_range():
    """Show the extreme range of electricity prices."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(-1500, 18500)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")

    # Full bar
    ax.barh(0.5, 18500, left=-1000, height=0.3, color="#eee", edgecolor="#ccc")

    # Annotated zones
    zones = [
        (-1000, 0, "#2196F3", "Negative\nprices"),
        (0, 100, "#4CAF50", "Normal\n($0–100)"),
        (100, 300, "#FFC107", "Elevated"),
        (300, 1000, "#FF9800", "Spikes"),
        (1000, 17500, "#F44336", "Extreme\nspikes"),
    ]
    for start, end, color, label in zones:
        width = end - start
        ax.barh(0.5, width, left=start, height=0.3, color=color, alpha=0.6, edgecolor="white")
        mid = start + width / 2
        if width > 800:
            ax.text(mid, 0.5, label, ha="center", va="center", fontsize=8, fontweight="bold")
        else:
            ax.text(mid, 1.0, label, ha="center", va="bottom", fontsize=7)
            ax.plot([mid, mid], [0.65, 0.95], color=color, linewidth=1)

    ax.text(-1000, 0.1, "-$1,000\n(floor)", ha="center", fontsize=8, color=BLUE)
    ax.text(17500, 0.1, "$17,500\n(cap)", ha="center", fontsize=8, color=RED)
    ax.text(50, 0.1, "$0", ha="center", fontsize=8, color=GRAY)

    ax.set_title("NEM price range ($/MWh) — most prices cluster in the green zone",
                 fontsize=11, fontweight="bold", y=1.3)
    save(fig, "01_price_range")


# ── CHAPTER 02 ──────────────────────────────────────────────

def ch02_duck_curve():
    """Illustrate the duck curve / diurnal price pattern."""
    hours = np.arange(0, 24, 0.5)
    # Stylised price profile
    price = 40 + 20 * np.sin((hours - 6) * np.pi / 12)
    # Solar dip (midday)
    solar_effect = -30 * np.exp(-0.5 * ((hours - 12) / 2) ** 2)
    # Evening spike
    evening = 60 * np.exp(-0.5 * ((hours - 18) / 1.5) ** 2)
    profile = price + solar_effect + evening
    profile = np.clip(profile, -10, 150)

    demand = 1200 + 400 * np.sin((hours - 6) * np.pi / 12)
    demand += 200 * np.exp(-0.5 * ((hours - 18) / 2) ** 2)
    solar = 800 * np.maximum(0, np.sin((hours - 6) * np.pi / 12)) * (hours > 6) * (hours < 18)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Price profile
    ax = axes[0]
    ax.fill_between(hours, profile, alpha=0.3, color=BLUE)
    ax.plot(hours, profile, color=BLUE, linewidth=2)
    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.set_ylabel("Typical price ($/MWh)", fontsize=10)
    ax.set_title("The 'duck curve' price profile", fontsize=11, fontweight="bold")

    # Annotations
    ax.annotate("Overnight\ntrough", xy=(3, 25), fontsize=8, ha="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.annotate("Solar dip\n(midday)", xy=(12, 10), fontsize=8, ha="center",
                color=ORANGE, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#fff3e0", alpha=0.8))
    ax.annotate("Evening\nspike!", xy=(18, 130), fontsize=8, ha="center",
                color=RED, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#ffebee", alpha=0.8))
    ax.set_xticks([0, 6, 12, 18, 24])

    # Demand vs solar
    ax = axes[1]
    ax.plot(hours, demand, color=BLUE, linewidth=2, label="Demand")
    ax.fill_between(hours, solar, alpha=0.3, color=ORANGE)
    ax.plot(hours, solar, color=ORANGE, linewidth=2, label="Solar generation")
    ax.plot(hours, demand - solar, color=RED, linewidth=2, linestyle="--", label="Net load")
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.set_ylabel("MW", fontsize=10)
    ax.set_title("Demand, solar, and net load", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xticks([0, 6, 12, 18, 24])

    fig.tight_layout()
    save(fig, "02_duck_curve")


def ch02_heavy_tails():
    """Compare Gaussian vs heavy-tailed distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    np.random.seed(42)
    gaussian = np.random.normal(50, 30, 10000)
    heavy = np.concatenate([
        np.random.normal(50, 25, 9500),
        np.random.exponential(200, 300) + 100,
        np.random.uniform(-200, 0, 200),
    ])

    for ax, data, label, color in [
        (axes[0], gaussian, "Gaussian distribution\n(normal — thin tails)", BLUE),
        (axes[1], heavy, "Electricity prices\n(heavy tails — extreme events)", RED),
    ]:
        ax.hist(data, bins=100, color=color, alpha=0.6, edgecolor="none", density=True)
        ax.set_xlabel("Price ($/MWh)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(-300, 600)

    # Annotate the tails
    axes[1].annotate("Spikes!\n(rare but huge)", xy=(400, 0.001), fontsize=9,
                     color=RED, fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="#ffebee"))
    axes[1].annotate("Negative\nprices", xy=(-150, 0.001), fontsize=9,
                     color=BLUE, fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="#e3f2fd"))

    fig.tight_layout()
    save(fig, "02_heavy_tails")


def ch02_acf_illustration():
    """Illustrate the concept of autocorrelation with a visual example."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    np.random.seed(42)
    hours = np.arange(0, 96)  # 2 days of half-hours

    # Generate a realistic price pattern
    base = 50 + 30 * np.sin((hours % 48 - 12) * np.pi / 24)
    noise = np.random.normal(0, 10, len(hours))
    prices = base + noise

    # Today vs yesterday overlay
    ax = axes[0, 0]
    today = prices[48:]
    yesterday = prices[:48]
    h = np.arange(48) / 2
    ax.plot(h, yesterday, color=GRAY, linewidth=2, label="Yesterday", alpha=0.7)
    ax.plot(h, today, color=BLUE, linewidth=2, label="Today")
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.set_ylabel("Price ($/MWh)", fontsize=10)
    ax.set_title("Today looks like yesterday\n(lag-48 autocorrelation)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Scatter plot: today vs yesterday
    ax = axes[0, 1]
    ax.scatter(yesterday, today, s=15, alpha=0.6, color=BLUE)
    ax.plot([0, 120], [0, 120], color=RED, linewidth=1, linestyle="--")
    ax.set_xlabel("Yesterday's price", fontsize=10)
    ax.set_ylabel("Today's price", fontsize=10)
    ax.set_title("Correlation at lag 48\n(strong positive relationship)", fontsize=10, fontweight="bold")

    # Stylised ACF
    ax = axes[1, 0]
    lags = np.arange(0, 200)
    acf = 0.3 * np.exp(-lags / 30) + 0.4 * np.exp(-((lags - 48) / 5)**2) + 0.2 * np.exp(-((lags - 96) / 5)**2) + 0.15 * np.exp(-((lags - 144) / 5)**2)
    acf[0] = 1.0
    ax.bar(lags, acf, width=1, color=BLUE, alpha=0.6)
    ax.axhline(0, color=GRAY, linewidth=0.5)
    for lag, label in [(48, "24h"), (96, "48h"), (144, "72h")]:
        ax.annotate(label, xy=(lag, acf[lag] + 0.02), fontsize=8, ha="center",
                   color=RED, fontweight="bold")
    ax.set_xlabel("Lag (half-hours)", fontsize=10)
    ax.set_ylabel("Autocorrelation", fontsize=10)
    ax.set_title("Autocorrelation function (ACF)\n(spikes every 48 lags = 24 hours)", fontsize=10, fontweight="bold")

    # Volatility clustering
    ax = axes[1, 1]
    np.random.seed(7)
    t = np.arange(300)
    vol = np.ones(300) * 10
    vol[80:120] = 80
    vol[200:230] = 60
    p = np.cumsum(np.random.normal(0, 1, 300) * vol) + 50
    ax.plot(t, p, linewidth=0.8, color=BLUE)
    ax.axhspan(p.min(), p.max(), xmin=80/300, xmax=120/300, alpha=0.1, color=RED)
    ax.axhspan(p.min(), p.max(), xmin=200/300, xmax=230/300, alpha=0.1, color=RED)
    ax.annotate("Volatile\nperiod", xy=(100, p[100]), fontsize=8, color=RED, fontweight="bold")
    ax.annotate("Quiet\nperiod", xy=(50, p[50]), fontsize=8, color=GREEN)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Price", fontsize=10)
    ax.set_title("Volatility clustering\n(calm and wild periods alternate)", fontsize=10, fontweight="bold")

    fig.tight_layout()
    save(fig, "02_acf_volatility")


def ch02_price_duration():
    """Price-duration curve illustration."""
    fig, ax = plt.subplots(figsize=(10, 5))

    np.random.seed(42)
    prices = np.concatenate([
        np.random.lognormal(3.5, 0.7, 9000),
        np.random.uniform(-100, 0, 500),
        np.random.lognormal(6, 1.5, 500),
    ])
    prices = np.clip(prices, -1000, 15000)
    sorted_p = np.sort(prices)[::-1]
    pct = np.linspace(0, 100, len(sorted_p))

    ax.plot(pct, sorted_p, color=BLUE, linewidth=1.5)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("% of time price is at or above this level", fontsize=11)
    ax.set_ylabel("Price ($/MWh)", fontsize=11)
    ax.set_title("Price-duration curve — where the money is", fontsize=12, fontweight="bold")

    # Shade zones
    ax.fill_between(pct[pct <= 5], sorted_p[pct <= 5], alpha=0.3, color=RED, label="Top 5% (spike zone)")
    ax.fill_between(pct[(pct >= 5) & (pct <= 50)], sorted_p[(pct >= 5) & (pct <= 50)],
                    alpha=0.15, color=ORANGE, label="5-50% (moderate)")
    ax.fill_between(pct[pct >= 50], sorted_p[pct >= 50], alpha=0.1, color=GREEN, label="Bottom 50% (cheap)")

    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, loc="upper right")

    ax.annotate("Battery DISCHARGES here\n(sell expensive electricity)",
                xy=(2, 2000), fontsize=9, color=RED, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#ffebee", alpha=0.8))
    ax.annotate("Battery CHARGES here\n(buy cheap electricity)",
                xy=(80, 10), fontsize=9, color=GREEN, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.8))

    save(fig, "02_price_duration")


# ── CHAPTER 03 ──────────────────────────────────────────────

def ch03_supply_stack():
    """Illustrate the merit-order supply stack."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    fuels = [
        ("Solar", 0, 400, "#FFD700"),
        ("Wind", 0, 600, "#2ECC71"),
        ("Hydro", 15, 200, "#1E90FF"),
        ("Coal", 35, 500, "#8B4513"),
        ("Gas\n(CCGT)", 70, 300, "#FF6B35"),
        ("Gas\n(OCGT)", 150, 200, "#FF4500"),
    ]

    x = 0
    for label, cost, cap, color in fuels:
        ax.fill_between([x, x + cap], 0, cost, alpha=0.4, color=color)
        ax.plot([x, x + cap, x + cap], [cost, cost, 0 if label == fuels[-1][0] else cost],
                color=color, linewidth=2)
        ax.text(x + cap / 2, cost + 8, label, ha="center", fontsize=9, fontweight="bold")
        ax.text(x + cap / 2, cost / 2 if cost > 20 else 5,
                f"${cost}", ha="center", fontsize=8, color="#333")
        x += cap

    # Demand line
    demand_level = 1200
    ax.axvline(demand_level, color=RED, linewidth=2, linestyle="--")
    ax.annotate("Demand\n(1,200 MW)", xy=(demand_level, 160), fontsize=10,
                color=RED, fontweight="bold", ha="left",
                bbox=dict(boxstyle="round", facecolor="#ffebee"))

    # Marginal price
    ax.axhline(70, color=RED, linewidth=1, linestyle=":", alpha=0.5)
    ax.annotate("Market price = $70\n(set by the LAST\ngenerator dispatched)",
                xy=(1600, 70), fontsize=9, color=RED,
                bbox=dict(boxstyle="round", facecolor="#ffebee"))

    ax.set_xlabel("Cumulative generation capacity (MW)", fontsize=11)
    ax.set_ylabel("Marginal cost ($/MWh)", fontsize=11)
    ax.set_title("The merit-order supply stack\n(cheapest generators dispatched first)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 200)
    ax.set_xlim(0, 2300)

    save(fig, "03_supply_stack")


def ch03_hockey_stick():
    """The net load vs price hockey stick."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    np.random.seed(42)
    net_load = np.linspace(-500, 2500, 500)
    # Hockey stick: flat then exponential
    price_base = np.where(net_load < 500, 10 + 0.02 * net_load,
                          10 + 0.02 * 500 + 0.0003 * (net_load - 500) ** 2)
    noise = np.random.normal(0, 15, len(net_load))
    prices = price_base + noise
    prices = np.clip(prices, -100, 5000)

    # Scatter
    ax = axes[0]
    ax.scatter(net_load, prices, s=2, alpha=0.3, color=BLUE)
    ax.plot(net_load, price_base, color=RED, linewidth=2, label="Underlying relationship")
    ax.set_xlabel("Net load (MW)", fontsize=10)
    ax.set_ylabel("Price ($/MWh)", fontsize=10)
    ax.set_title("The 'hockey stick'\n(net load vs price)", fontsize=11, fontweight="bold")
    ax.set_yscale("symlog", linthresh=100)
    ax.axvline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")

    # Annotated zones
    ax.annotate("FLAT: renewables\nsatisfy demand", xy=(-200, 15), fontsize=8,
                color=GREEN, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#e8f5e9"))
    ax.annotate("STEEP: gas peakers\nset the price", xy=(2000, 500), fontsize=8,
                color=RED, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#ffebee"))

    # Diagram explanation
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("What is net load?", fontsize=11, fontweight="bold")

    boxes = [
        (5, 8.5, "Total electricity\nDEMAND", BLUE, 3.5),
        (2, 5.5, "Wind\ngeneration", GREEN, 2.5),
        (6.5, 5.5, "Solar\ngeneration", ORANGE, 2.5),
        (5, 2, "NET LOAD\n= what's left for\ngas & coal", RED, 4),
    ]
    for x, y, text, color, w in boxes:
        rect = mpatches.FancyBboxPatch((x - w/2, y - 0.7), w, 1.4,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    ax.annotate("", xy=(5, 7.5), xytext=(5, 7.8), arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.annotate("minus", xy=(3.2, 7), fontsize=10, color=GRAY, fontweight="bold")
    ax.annotate("", xy=(2, 4.5), xytext=(3.5, 3), arrowprops=dict(arrowstyle="<-", color=GRAY, lw=2))
    ax.annotate("", xy=(6.5, 4.5), xytext=(5.5, 3), arrowprops=dict(arrowstyle="<-", color=GRAY, lw=2))
    ax.text(5, 6.5, "=", fontsize=16, ha="center", color=GRAY, fontweight="bold")

    fig.tight_layout()
    save(fig, "03_hockey_stick")


def ch03_spike_anatomy():
    """Anatomy of a price spike — timeline."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    hours = np.arange(0, 24, 0.5)

    # Price
    price = 50 + 20 * np.sin((hours - 6) * np.pi / 12)
    spike = 800 * np.exp(-0.5 * ((hours - 18) / 0.5) ** 2)
    price = price + spike
    axes[0].plot(hours, price, color=RED, linewidth=2)
    axes[0].fill_between(hours, price, alpha=0.2, color=RED)
    axes[0].set_ylabel("Price ($/MWh)", fontsize=10)
    axes[0].set_title("Anatomy of a price spike", fontsize=12, fontweight="bold")
    axes[0].axhline(300, color=GRAY, linewidth=1, linestyle="--")
    axes[0].annotate("Spike\nthreshold", xy=(22, 300), fontsize=8, color=GRAY)

    # Phases
    for x, w, label, color in [
        (14, 3, "1. BUILD-UP\nSolar drops,\ngas ramps", ORANGE),
        (17, 2, "2. SPIKE\nDemand > supply,\nprice explodes", RED),
        (19, 3, "3. RESPONSE\nBatteries discharge,\ndemand drops", GREEN),
    ]:
        axes[0].axvspan(x, x+w, alpha=0.1, color=color)
        axes[0].text(x + w/2, 600, label, fontsize=7, ha="center", color=color, fontweight="bold")

    # Solar and wind
    solar = 800 * np.maximum(0, np.sin((hours - 6) * np.pi / 12)) * (hours > 6) * (hours < 18.5)
    wind = 200 + 100 * np.sin(hours * 0.3) + np.random.RandomState(42).normal(0, 30, len(hours))
    wind = np.clip(wind, 50, 500)
    axes[1].fill_between(hours, solar, alpha=0.4, color=ORANGE, label="Solar")
    axes[1].fill_between(hours, wind, alpha=0.4, color=GREEN, label="Wind")
    axes[1].set_ylabel("Generation (MW)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].annotate("Solar cliff!\n(sunset)", xy=(17, 100), fontsize=9,
                     color=ORANGE, fontweight="bold")

    # Net load
    demand = 1200 + 400 * np.sin((hours - 6) * np.pi / 12) + 200 * np.exp(-((hours-18)/2)**2)
    nl = demand - solar - wind
    axes[2].plot(hours, nl, color=PURPLE, linewidth=2)
    axes[2].fill_between(hours, nl, alpha=0.2, color=PURPLE)
    axes[2].set_ylabel("Net load (MW)", fontsize=10)
    axes[2].set_xlabel("Hour of day", fontsize=10)
    axes[2].axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    axes[2].annotate("Net load peaks\nas solar dies", xy=(18, nl[36]),
                     fontsize=9, color=PURPLE, fontweight="bold")

    for ax in axes:
        ax.set_xticks([0, 6, 12, 18, 24])
    fig.tight_layout()
    save(fig, "03_spike_anatomy")


# ── CHAPTER 04 ──────────────────────────────────────────────

def ch04_clear_sky_index():
    """Clear-sky index concept."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    hours = np.linspace(5, 19, 100)

    clearsky = 800 * np.sin((hours - 5) * np.pi / 14)
    clearsky = np.maximum(clearsky, 0)

    # Cloudy day
    cloud = clearsky * (0.3 + 0.4 * np.sin(hours * 2) ** 2)
    cloud = np.clip(cloud, 0, 900)

    ax = axes[0]
    ax.fill_between(hours, clearsky, alpha=0.2, color=ORANGE)
    ax.plot(hours, clearsky, color=ORANGE, linewidth=2, linestyle="--", label="Clear-sky (theoretical max)")
    ax.plot(hours, cloud, color=BLUE, linewidth=2, label="Actual (cloudy day)")
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.set_ylabel("Solar irradiance (W/m²)", fontsize=10)
    ax.set_title("Clear-sky model vs actual irradiance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.annotate("Gap = clouds\nblocking sunlight", xy=(12, 400), fontsize=9,
                color=RED, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#ffebee"))

    # CSI
    csi = np.where(clearsky > 10, cloud / clearsky, np.nan)
    ax = axes[1]
    ax.plot(hours, csi, color=GREEN, linewidth=2)
    ax.axhline(1.0, color=ORANGE, linewidth=1, linestyle="--", label="Perfect clear sky (CSI=1)")
    ax.axhline(0.5, color=GRAY, linewidth=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.set_ylabel("Clear-sky index (CSI)", fontsize=10)
    ax.set_title("Clear-sky index\n(removes time-of-day effect)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=8)
    ax.annotate("CSI < 1 = cloudy", xy=(10, 0.4), fontsize=9, color=BLUE)
    ax.annotate("CSI ≈ 1 = clear", xy=(16, 1.05), fontsize=9, color=ORANGE)

    fig.tight_layout()
    save(fig, "04_clear_sky_index")


def ch04_wind_power_curve():
    """Wind turbine power curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ws = np.linspace(0, 30, 300)
    power = np.zeros_like(ws)
    # Cut-in at 3 m/s
    mask1 = (ws >= 3) & (ws < 12)
    power[mask1] = 3000 * ((ws[mask1] - 3) / 9) ** 3
    # Rated 12-25
    mask2 = (ws >= 12) & (ws < 25)
    power[mask2] = 3000
    # Cut-out above 25
    mask3 = ws >= 25
    power[mask3] = 0

    ax.plot(ws, power / 1000, color=GREEN, linewidth=2.5)
    ax.fill_between(ws, power / 1000, alpha=0.15, color=GREEN)

    # Annotate zones
    zones = [
        (1.5, "No output\n(too calm)", "#eee", GRAY),
        (7.5, "Cubic zone\nP ∝ v³", "#e8f5e9", GREEN),
        (18, "Rated output\n(pitch control)", "#fff3e0", ORANGE),
        (27, "Shutdown\n(too windy!)", "#ffebee", RED),
    ]
    for x, text, bg, color in zones:
        ax.annotate(text, xy=(x, 3.5), fontsize=9, ha="center",
                   color=color, fontweight="bold",
                   bbox=dict(boxstyle="round", facecolor=bg))

    ax.axvline(3, color=GRAY, linewidth=1, linestyle=":", alpha=0.5)
    ax.axvline(12, color=GRAY, linewidth=1, linestyle=":", alpha=0.5)
    ax.axvline(25, color=GRAY, linewidth=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Wind speed (m/s)", fontsize=11)
    ax.set_ylabel("Power output (MW)", fontsize=11)
    ax.set_title("Wind turbine power curve\n(small wind changes → big power changes in the cubic zone)",
                 fontsize=12, fontweight="bold")

    save(fig, "04_wind_power_curve")


# ── CHAPTER 05 ──────────────────────────────────────────────

def ch05_rolling_origin():
    """Rolling-origin backtest diagram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 6)
    ax.set_title("Rolling-origin backtest with embargo", fontsize=12, fontweight="bold")

    for fold, y in enumerate([4.5, 3, 1.5]):
        # Training
        train_end = 4 + fold * 1.5
        ax.barh(y, train_end, left=0, height=0.5, color=BLUE, alpha=0.4)
        ax.text(train_end / 2, y, "TRAIN", ha="center", va="center", fontsize=8, fontweight="bold", color=BLUE)

        # Embargo
        ax.barh(y, 1, left=train_end, height=0.5, color=GRAY, alpha=0.2)
        ax.text(train_end + 0.5, y, "gap", ha="center", va="center", fontsize=7, color=GRAY)

        # Test
        ax.barh(y, 1.5, left=train_end + 1, height=0.5, color=GREEN, alpha=0.4)
        ax.text(train_end + 1.75, y, "TEST", ha="center", va="center", fontsize=8, fontweight="bold", color=GREEN)

        ax.text(-0.3, y, f"Fold {fold+1}", fontsize=9, va="center", ha="right")

    # Labels
    ax.annotate("Training data grows →", xy=(6, 5.3), fontsize=9, color=BLUE)
    ax.annotate("Embargo prevents\ninformation leakage", xy=(8, -0.3), fontsize=9,
                color=GRAY, ha="center",
                bbox=dict(boxstyle="round", facecolor="#f5f5f5"))
    ax.annotate("", xy=(0, 0.5), xytext=(12, 0.5),
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.text(12.2, 0.5, "time →", fontsize=9, color=GRAY, va="center")

    save(fig, "05_rolling_origin")


def ch05_battery_dispatch():
    """Battery LP dispatch illustration."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    hours = np.arange(0, 24, 0.5)
    price = 40 + 20 * np.sin((hours - 6) * np.pi / 12)
    solar_dip = -25 * np.exp(-0.5 * ((hours - 12) / 2) ** 2)
    evening = 50 * np.exp(-0.5 * ((hours - 18) / 1.5) ** 2)
    price = price + solar_dip + evening
    price = np.clip(price, -5, 130)

    # Optimal schedule (simplified)
    charge = np.zeros(48)
    discharge = np.zeros(48)
    charge[(hours >= 10) & (hours <= 14)] = 80  # Charge during solar dip
    discharge[(hours >= 17) & (hours <= 20)] = 80  # Discharge during peak

    ax = axes[0]
    ax.plot(hours, price, color=BLUE, linewidth=2)
    ax.fill_between(hours, price, where=charge > 0, alpha=0.3, color=GREEN, label="Charging (buying cheap)")
    ax.fill_between(hours, price, where=discharge > 0, alpha=0.3, color=RED, label="Discharging (selling expensive)")
    ax.set_ylabel("Price ($/MWh)", fontsize=10)
    ax.set_title("Battery arbitrage: buy low, sell high", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # SOC
    soc = np.zeros(49)
    for t in range(48):
        soc[t+1] = soc[t] + charge[t] * 0.92 * 0.5 - discharge[t] / 0.92 * 0.5
        soc[t+1] = np.clip(soc[t+1], 0, 200)

    ax = axes[1]
    ax.fill_between(hours, soc[:48], alpha=0.3, color=PURPLE)
    ax.plot(hours, soc[:48], color=PURPLE, linewidth=2)
    ax.set_ylabel("State of charge (MWh)", fontsize=10)
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.set_title("Battery state of charge", fontsize=11, fontweight="bold")
    ax.annotate("Fills up during\ncheap midday", xy=(12, 120), fontsize=9,
                color=GREEN, fontweight="bold")
    ax.annotate("Empties during\nexpensive evening", xy=(19, 30), fontsize=9,
                color=RED, fontweight="bold")

    for ax in axes:
        ax.set_xticks([0, 6, 12, 18, 24])
    fig.tight_layout()
    save(fig, "05_battery_dispatch")


def ch05_capture_ratio():
    """Capture ratio concept."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Perfect\nforesight", "Good\nforecast", "Mediocre\nforecast", "Naive\n(no forecast)"]
    revenues = [100, 72, 45, 20]
    colors = ["#333", GREEN, ORANGE, RED]

    bars = ax.bar(categories, revenues, color=colors, alpha=0.7, edgecolor="white", linewidth=2)

    for bar, rev in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width() / 2, rev + 2,
                f"CR = {rev/100:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, rev / 2,
                f"${rev}K", ha="center", fontsize=12, fontweight="bold", color="white")

    ax.set_ylabel("Annual revenue ($K)", fontsize=11)
    ax.set_title("Capture ratio = your revenue / maximum possible revenue",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 120)

    save(fig, "05_capture_ratio")


# ── CHAPTER 06 ──────────────────────────────────────────────

def ch06_lasso():
    """LASSO regularisation concept."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before LASSO — all features have coefficients
    ax = axes[0]
    features = ["lag_48", "lag_96", "lag_336", "hour", "dow", "month",
                "demand", "temp", "wind", "solar", "feat11", "feat12"]
    coefs = [0.6, 0.15, 0.25, 0.3, 0.1, 0.05, 0.2, 0.08, 0.03, 0.02, 0.01, 0.005]
    colors_b = [BLUE] * len(features)
    ax.barh(features, coefs, color=colors_b, alpha=0.6)
    ax.set_xlabel("Coefficient magnitude", fontsize=10)
    ax.set_title("Before LASSO\n(all features kept, noise included)", fontsize=11, fontweight="bold")

    # After LASSO — sparse
    ax = axes[1]
    coefs_lasso = [0.65, 0, 0.28, 0.35, 0.12, 0, 0.22, 0, 0, 0, 0, 0]
    colors_a = [GREEN if c > 0 else GRAY for c in coefs_lasso]
    ax.barh(features, coefs_lasso, color=colors_a, alpha=0.6)
    ax.set_xlabel("Coefficient magnitude", fontsize=10)
    ax.set_title("After LASSO\n(irrelevant features zeroed out)", fontsize=11, fontweight="bold")

    for i, c in enumerate(coefs_lasso):
        if c == 0:
            ax.text(0.01, i, "× removed", fontsize=7, color=RED, va="center")

    fig.tight_layout()
    save(fig, "06_lasso")


# ── CHAPTER 07 ──────────────────────────────────────────────

def ch07_boosting():
    """Gradient boosting illustration."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = np.sin(x) + 0.5 * np.sin(3 * x)
    noise = np.random.normal(0, 0.2, len(x))
    y = y_true + noise

    preds = [np.zeros_like(x)]
    for i in range(3):
        residual = y - preds[-1]
        # Simple tree approximation
        from numpy.polynomial import polynomial as P
        c = P.polyfit(x, residual, 3 + i * 2)
        new_pred = P.polyval(x, c) * 0.3
        preds.append(preds[-1] + new_pred)

    titles = ["Data + Tree 1", "+ Tree 2", "+ Tree 3", "Final (sum of trees)"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.scatter(x, y, s=10, color=GRAY, alpha=0.5)
        ax.plot(x, y_true, color=GRAY, linewidth=1, linestyle="--", alpha=0.3)
        ax.plot(x, preds[i + 1] if i < 3 else preds[-1], color=BLUE if i < 3 else GREEN, linewidth=2)
        if i > 0 and i < 3:
            ax.plot(x, preds[i], color=ORANGE, linewidth=1, alpha=0.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(-2, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Gradient boosting: each tree fixes the previous trees' mistakes",
                 fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "07_boosting")


def ch07_quantile_fan():
    """Quantile fan / prediction intervals."""
    fig, ax = plt.subplots(figsize=(10, 5))

    hours = np.arange(0, 48)
    median = 50 + 20 * np.sin((hours - 12) * np.pi / 24) + 30 * np.exp(-((hours - 36) / 3) ** 2)

    spreads = [(5, "#2c5f8a", "50% interval (q25–q75)"),
               (15, "#5b8db8", "80% interval (q10–q90)"),
               (30, "#a3c4e0", "90% interval (q05–q95)")]

    for spread, color, label in reversed(spreads):
        noise = spread * (1 + 0.5 * np.sin(hours * 0.3))
        ax.fill_between(hours, median - noise, median + noise, alpha=0.4, color=color, label=label)

    ax.plot(hours, median, color="white", linewidth=3)
    ax.plot(hours, median, color=BLUE, linewidth=2, label="Median forecast (q50)")

    # Fake actual
    np.random.seed(42)
    actual = median + np.random.normal(0, 12, len(hours))
    actual[36] = median[36] + 60  # A surprise spike
    ax.scatter(hours, actual, s=15, color=RED, zorder=5, label="Actual prices")

    ax.annotate("Actual spike falls\noutside 80% interval!", xy=(36, actual[36]),
                xytext=(40, actual[36] + 20), fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED),
                bbox=dict(boxstyle="round", facecolor="#ffebee"))

    ax.set_xlabel("Half-hour period", fontsize=10)
    ax.set_ylabel("Price ($/MWh)", fontsize=10)
    ax.set_title("Quantile fan: showing what the model thinks is likely",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    save(fig, "07_quantile_fan")


# ── CHAPTER 08 ──────────────────────────────────────────────

def ch08_calibration():
    """Reliability diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    quantiles = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

    # Well-calibrated
    ax = axes[0]
    observed_good = quantiles + np.random.RandomState(42).normal(0, 0.02, len(quantiles))
    observed_good = np.clip(observed_good, 0, 1)
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.scatter(quantiles, observed_good, s=80, color=GREEN, zorder=5)
    ax.plot(quantiles, observed_good, color=GREEN, linewidth=2, label="Well-calibrated model")
    ax.set_xlabel("Nominal quantile level", fontsize=10)
    ax.set_ylabel("Observed frequency", fontsize=10)
    ax.set_title("Well-calibrated\n(points on the diagonal)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Overconfident
    ax = axes[1]
    observed_bad = quantiles * 0.7 + 0.15
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.scatter(quantiles, observed_bad, s=80, color=RED, zorder=5)
    ax.plot(quantiles, observed_bad, color=RED, linewidth=2, label="Overconfident model")
    ax.fill_between(quantiles, quantiles, observed_bad, alpha=0.1, color=RED)
    ax.set_xlabel("Nominal quantile level", fontsize=10)
    ax.set_ylabel("Observed frequency", fontsize=10)
    ax.set_title("Overconfident\n(intervals too narrow → misses extremes)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate("Says 10% will be below,\nbut actually 25% are",
                xy=(0.1, 0.25), xytext=(0.3, 0.5), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED),
                bbox=dict(boxstyle="round", facecolor="#ffebee"))

    fig.tight_layout()
    save(fig, "08_calibration")


# ── CHAPTER 09 ──────────────────────────────────────────────

def ch09_scenario_vs_cc():
    """Scenario MPC vs chance-constrained MPC."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    hours = np.arange(0, 24, 0.5)
    median = 50 + 20 * np.sin((hours - 6) * np.pi / 12) + 40 * np.exp(-((hours-18)/1.5)**2)

    # Scenario MPC
    ax = axes[0]
    np.random.seed(42)
    for i in range(15):
        scenario = median + np.random.normal(0, 15, len(hours))
        ax.plot(hours, scenario, linewidth=0.8, alpha=0.3, color=BLUE)
    ax.plot(hours, median, color=RED, linewidth=2, label="Median")
    ax.set_xlabel("Hour", fontsize=10)
    ax.set_ylabel("Price ($/MWh)", fontsize=10)
    ax.set_title("Scenario MPC\n(sample many paths, optimise average)",
                 fontsize=11, fontweight="bold")
    ax.annotate("Each line = one\npossible future", xy=(6, 80), fontsize=9,
                color=BLUE, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#e3f2fd"))
    ax.legend(fontsize=8)
    ax.set_xticks([0, 6, 12, 18, 24])

    # Chance-constrained
    ax = axes[1]
    q_lo = median - 20
    q_hi = median + 20
    ax.fill_between(hours, q_lo, q_hi, alpha=0.3, color=BLUE, label="80% interval")
    ax.plot(hours, median, color=BLUE, linewidth=2)
    ax.plot(hours, q_lo, color=RED, linewidth=1.5, linestyle="--", label="Conservative price (low)")
    ax.plot(hours, q_hi, color=GREEN, linewidth=1.5, linestyle="--", label="Conservative cost (high)")
    ax.set_xlabel("Hour", fontsize=10)
    ax.set_ylabel("Price ($/MWh)", fontsize=10)
    ax.set_title("Chance-constrained MPC\n(use pessimistic bounds)",
                 fontsize=11, fontweight="bold")
    ax.annotate("Only discharge if\nprofitable even at\nthe LOW price", xy=(18, q_lo[36]),
                fontsize=8, color=RED, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#ffebee"))
    ax.legend(fontsize=8)
    ax.set_xticks([0, 6, 12, 18, 24])

    fig.tight_layout()
    save(fig, "09_scenario_vs_cc")


def ch09_value_of_info():
    """Value of information curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mae = np.array([0, 10, 20, 30, 50, 80, 120, 200])
    cr = np.array([1.0, 0.85, 0.72, 0.62, 0.48, 0.35, 0.22, 0.10])

    ax.plot(mae, cr, "o-", color=BLUE, linewidth=2, markersize=8)
    ax.fill_between(mae, cr, alpha=0.1, color=BLUE)

    ax.set_xlabel("Forecast error — MAE ($/MWh)", fontsize=11)
    ax.set_ylabel("Capture ratio", fontsize=11)
    ax.set_title("Value of forecast quality\n(concave — first improvements worth the most)",
                 fontsize=12, fontweight="bold")

    ax.annotate("Perfect forecast", xy=(0, 1.0), xytext=(30, 0.95),
                fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("Diminishing returns\nhere", xy=(80, 0.35), fontsize=9,
                color=ORANGE, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#fff3e0"))
    ax.annotate("Most value\ngained here!", xy=(15, 0.78), fontsize=9,
                color=RED, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#ffebee"))

    ax.set_ylim(0, 1.1)
    save(fig, "09_value_of_info")


# ── CHAPTER 10 ──────────────────────────────────────────────

def ch10_pipeline():
    """End-to-end pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)

    stages = [
        (1, "Weather\nData", BLUE, "ERA5 reanalysis"),
        (3.5, "Feature\nEngineering", GREEN, "CSI, wind, lags"),
        (6, "Price\nModel", ORANGE, "GBT / LEAR / QRA"),
        (8.5, "Calibration", PURPLE, "Conformal prediction"),
        (11, "Dispatch\nLP", RED, "cvxpy optimisation"),
        (13.5, "Revenue", "#333", "Capture ratio"),
    ]

    for x, label, color, detail in stages:
        rect = mpatches.FancyBboxPatch((x - 0.9, 1.2), 1.8, 1.6,
                                       boxstyle="round,pad=0.15",
                                       facecolor=color, alpha=0.15,
                                       edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 2.3, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color=color)
        ax.text(x, 1.5, detail, ha="center", va="center", fontsize=7, color=GRAY)

    # Arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.9
        x2 = stages[i + 1][0] - 0.9
        ax.annotate("", xy=(x2, 2), xytext=(x1, 2),
                     arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=2))

    ax.set_title("The complete pipeline: from sunlight to revenue",
                 fontsize=13, fontweight="bold", y=1.05)

    # Error labels
    for i, (x, _, _, _) in enumerate(stages[:-1]):
        mid = (x + stages[i + 1][0]) / 2
        ax.text(mid, 0.8, "error\ncompounds →", ha="center", fontsize=7,
                color=RED, alpha=0.5)

    save(fig, "10_pipeline")


def ch10_greybox():
    """Grey-box model architecture."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Input
    rect = mpatches.FancyBboxPatch((0.5, 3), 2, 2, boxstyle="round,pad=0.15",
                                   facecolor=BLUE, alpha=0.15, edgecolor=BLUE, linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 4.3, "Net Load", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(1.5, 3.7, "(demand − renewables)", ha="center", fontsize=8, color=GRAY)

    # White box
    rect = mpatches.FancyBboxPatch((4, 5), 3, 2, boxstyle="round,pad=0.15",
                                   facecolor=ORANGE, alpha=0.15, edgecolor=ORANGE, linewidth=2)
    ax.add_patch(rect)
    ax.text(5.5, 6.3, "WHITE BOX", ha="center", fontsize=10, fontweight="bold", color=ORANGE)
    ax.text(5.5, 5.7, "Merit-order curve\n(isotonic regression)", ha="center", fontsize=8, color=GRAY)
    ax.text(5.5, 5.2, "Physics-based", ha="center", fontsize=7, color=ORANGE, style="italic")

    # Black box
    rect = mpatches.FancyBboxPatch((4, 1.5), 3, 2, boxstyle="round,pad=0.15",
                                   facecolor=PURPLE, alpha=0.15, edgecolor=PURPLE, linewidth=2)
    ax.add_patch(rect)
    ax.text(5.5, 2.8, "BLACK BOX", ha="center", fontsize=10, fontweight="bold", color=PURPLE)
    ax.text(5.5, 2.2, "GBT on residuals\n(lags, calendar, weather)", ha="center", fontsize=8, color=GRAY)
    ax.text(5.5, 1.7, "Data-driven", ha="center", fontsize=7, color=PURPLE, style="italic")

    # Output
    rect = mpatches.FancyBboxPatch((9, 3), 2.5, 2, boxstyle="round,pad=0.15",
                                   facecolor=GREEN, alpha=0.15, edgecolor=GREEN, linewidth=2)
    ax.add_patch(rect)
    ax.text(10.25, 4.3, "Price forecast", ha="center", fontsize=11, fontweight="bold", color=GREEN)
    ax.text(10.25, 3.7, "= white + black", ha="center", fontsize=9, color=GRAY)

    # Arrows
    ax.annotate("", xy=(4, 6), xytext=(2.5, 4.5), arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.annotate("", xy=(4, 2.5), xytext=(2.5, 3.5), arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.annotate("", xy=(9, 4.3), xytext=(7, 5.8), arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.annotate("", xy=(9, 3.7), xytext=(7, 2.7), arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.text(8, 5.3, "+", fontsize=16, ha="center", color=GRAY, fontweight="bold")

    ax.set_title("Grey-box model: best of both worlds", fontsize=13, fontweight="bold")

    save(fig, "10_greybox")


if __name__ == "__main__":
    print("Generating figures...")
    # Ch 01
    ch01_nem_regions()
    ch01_interval_timestamps()
    ch01_arcsinh_transform()
    ch01_price_range()
    # Ch 02
    ch02_duck_curve()
    ch02_heavy_tails()
    ch02_acf_illustration()
    ch02_price_duration()
    # Ch 03
    ch03_supply_stack()
    ch03_hockey_stick()
    ch03_spike_anatomy()
    # Ch 04
    ch04_clear_sky_index()
    ch04_wind_power_curve()
    # Ch 05
    ch05_rolling_origin()
    ch05_battery_dispatch()
    ch05_capture_ratio()
    # Ch 06
    ch06_lasso()
    # Ch 07
    ch07_boosting()
    ch07_quantile_fan()
    # Ch 08
    ch08_calibration()
    # Ch 09
    ch09_scenario_vs_cc()
    ch09_value_of_info()
    # Ch 10
    ch10_pipeline()
    ch10_greybox()
    print("Done!")
