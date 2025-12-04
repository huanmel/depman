#!/usr/bin/env python3
"""
Depman: Gitman CLI wrapper + update checker.

Usage:
    pip install -e .
    depman check_updates --recursive
"""
    # depman gm install  # Wraps gitman install

import click
from pathlib import Path
from typing import Optional
from .commands.gm_commands import gm
from .commands.checker import check_cmd, list_cmd
from .utils.configs import find_git_root




CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)

@click.option("--root", type=str, default=None, help="Git root path (auto-detects).")
# def cli(root: Optional[str]):
#     pass
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
cli.add_command(check_cmd)

cli.add_command(list_cmd)


if __name__ == "__main__":
    cli()