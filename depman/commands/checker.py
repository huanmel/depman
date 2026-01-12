"""
Update checker using Gitman API + GitPython.
"""

from pathlib import Path

import click
import yaml  # For manual YAML loads
from git import GitCommandError, Repo
from gitman.models import Config
from rich.console import Console
from rich.table import Table

from depman import CACHE_CONFIGS, CACHE_GIT_REPOS, CONFIG_NAME
from depman.utils.configs import (
    get_configs_and_repos,
    print_check_table,
    print_list_configs_repos,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
# @click.option("--recursive", is_flag=True, help="Flatten deps recursively for updates (legacy).")
# @click.option("--scan-depth", type=int, default=None, help="Max recursion depth for scanning (default: unlimited).")
@click.option(
    "--use-cache",
    "-c",
    is_flag=True,
    default=False,
    help="use cached YAML files instead of live scan (path to root).",
)
@click.option(
    "--update-mode",
    "-u",
    is_flag=True,
    default=False,
    help="update projects",
)
@click.option(
    "--list-mode",
    "-l",
    is_flag=True,
    default=False,
    help="list mode: number projects to be opened",
)
@click.option(
    "--terminal-mode",
    "-t",
    is_flag=True,
    default=False,
    help="only in list mode: open selected project in terminal.",
)
@click.pass_context
# , recursive: bool, scan_depth: Optional[int]
def check_cmd(ctx: click.Context, use_cache: bool,update_mode: bool,list_mode: bool,terminal_mode: bool):
    """Check for upstream updates in Gitman dependencies (now scans all projects)."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache)
    print_check_table(loaded_configs, git_repos, root=root,update_mode=update_mode,list_mode=list_mode,list_open_terminal=terminal_mode)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--use_cache",
    "-c",
    is_flag=True,
    default=False,
    help="use cached YAML files instead of live scan (path to root).",
)
@click.option("--dirty", "-d", is_flag=True, default=False, help="print only dirty.")
@click.pass_context
def list_cmd(ctx: click.Context, use_cache: bool, dirty: bool):
    """List all Git/Gitman projects under root."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache)
    print_list_configs_repos(loaded_configs, git_repos, only_dirty=dirty)


if __name__ == "__main__":
    from depman.cli import cli

    root = r"C:\Users\ivanm\Documents\MATLAB\EKL\deps_tests\gitman_proj_test"
    # cli(['--root', root, 'check-updates'])
    cli(["--root", root, "list", "-c", "-d"])
