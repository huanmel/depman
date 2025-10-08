#!/usr/bin/env python3
"""
Depman: Gitman CLI wrapper + update checker.

Usage:
    pip install -e .
    depman gm install  # Wraps gitman install
    depman check_updates --recursive
"""

import click
from pathlib import Path
from typing import Optional
from .commands import gm
from .checker import check_updates_cmd


def find_git_root(start: str = ".") -> Optional[str]:
    """Find the nearest Git root."""
    start_path = Path(start).resolve()
    current = start_path
    while current != current.parent:
        if (current / ".git").exists():
            return str(current)
        current = current.parent
    return None


@click.group()
@click.option("--root", type=str, default=None, help="Git root path (auto-detects).")
@click.pass_context
def cli(ctx: click.Context, root: Optional[str]):
    """Depman: Enhance Gitman dependency management."""
    if root is None:
        root = find_git_root()
        if root is None:
            raise click.ClickException("No Git root found. Run from a Git repo.")
    ctx.ensure_object(dict)
    ctx.obj["root"] = Path(root)


cli.add_command(gm)
cli.add_command(check_updates_cmd)


if __name__ == "__main__":
    cli()