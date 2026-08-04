"""Load and validate the project configuration from config.yaml."""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    """Read the YAML config file and return it as a dict.

    Args:
        path: Path to config file. Defaults to ``config.yaml`` at the repo root.

    Returns:
        Parsed configuration dictionary.
    """
    path = path or _DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return _REPO_ROOT
