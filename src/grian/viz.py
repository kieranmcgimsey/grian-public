"""Shared plotting style and figure-saving utility.

All plot *construction* happens in notebook cells so figures can be
tweaked during exploration. This module provides only the shared style
and a save helper.
"""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURE_DIR = Path("outputs/figures")

STYLE = {
    "figure.figsize": (12, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}


def apply_style() -> None:
    """Apply the shared matplotlib style to the current session."""
    plt.rcParams.update(STYLE)


def save_fig(fig: plt.Figure, name: str, caption: str = "") -> Path:
    """Save a figure to the shared output directory.

    Args:
        fig: Matplotlib figure to save.
        name: Filename (without extension).
        caption: Optional caption stored as figure metadata.

    Returns:
        Path to the saved PNG file.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    if caption:
        fig.text(
            0.5, -0.02, caption,
            ha="center", fontsize=9, style="italic",
        )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
