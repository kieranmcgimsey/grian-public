"""Shared helpers for exposing modules as Fire CLIs.

Every substantive module ends with a ``main()`` that calls
:func:`run_module_cli`, so ``python -m grian.<module>`` turns that module's
public, locally-defined callables into a command-line tool with no per-file
boilerplate. The curated top-level tool lives in :mod:`grian.cli`.
"""

from __future__ import annotations

from typing import Any


def run_module_cli(namespace: dict[str, Any]) -> None:
    """Expose a module's public, locally-defined callables as a Fire CLI.

    Args:
        namespace: The module's ``globals()``. Names that are private
            (leading underscore) or imported from elsewhere are hidden, so the
            CLI shows only what the module itself defines.
    """
    import fire

    module_name = namespace.get("__name__")
    components = {
        name: obj
        for name, obj in namespace.items()
        if not name.startswith("_")
        and callable(obj)
        and getattr(obj, "__module__", None) == module_name
    }
    # Fall back to the whole namespace for modules whose public surface is data
    # (e.g. model spec dicts) rather than functions.
    fire.Fire(components or namespace)


def console():
    """Return a shared Rich console (stderr-safe, colour-aware)."""
    from rich.console import Console

    return Console()
