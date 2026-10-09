#!/usr/bin/env python3
"""
Depman: Gitman CLI wrapper + update checker.

Usage:
    pip install -e .
    depman check_updates --recursive
"""
    # depman gm install  # Wraps gitman install

import click
import typer
from pathlib import Path
from typing import Optional
from .commands.gm_commands import gm_app, gm_lock
from .commands.checker import check_cmd, list_cmd, CONTEXT_SETTINGS as CHECKER_CONTEXT_SETTINGS
from .commands.review import review_cmd, CONTEXT_SETTINGS as REVIEW_CONTEXT_SETTINGS
from .commands.hooks import hooks_app
from .utils.configs import find_git_root


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

app = typer.Typer(
    context_settings=CONTEXT_SETTINGS,
    help="Depman: Enhance Gitman dependency management.",
)


@app.callback()
def main(ctx: typer.Context, root: Optional[str] = typer.Option(None, "--root", help="Git root path (auto-detects).")):
    """Depman: Enhance Gitman dependency management."""
    if root is None:
        root = find_git_root()
        if root is None:
            raise click.ClickException("No Git root found. Run from a Git repo.")
    ctx.obj = {"root": Path(root)}


app.add_typer(gm_app)
app.command("lock", help="Lock dependencies (alias of `gm lock`).")(gm_lock)
app.command("check", context_settings=CHECKER_CONTEXT_SETTINGS)(check_cmd)
app.command("list", context_settings=CHECKER_CONTEXT_SETTINGS)(list_cmd)
app.command("review", context_settings=REVIEW_CONTEXT_SETTINGS)(review_cmd)
app.add_typer(hooks_app)

# Exposed as an ordinary click.Group so the `depman = "depman.cli:cli"` entry point
# in pyproject.toml and CliRunner-based tests keep invoking it exactly as before.
cli = typer.main.get_command(app)


if __name__ == "__main__":
    app()
