"""
Update checker using Gitman API + GitPython.
"""

import json

import click
import typer

from depman.commands.review import ORDER_CHOICES, ORDER_HELP, run_review
from depman.utils.configs import (
    get_configs_and_repos,
    print_check_table,
    print_list_configs_repos,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def check_cmd(
    ctx: typer.Context,
    use_cache: bool = typer.Option(
        False, "--use-cache", "-c",
        help="use cached YAML files instead of live scan (path to root).",
    ),
    update_mode: bool = typer.Option(
        False, "--update-mode", "-u",
        help="update projects",
    ),
    list_mode: bool = typer.Option(
        False, "--list-mode", "-l",
        help="list mode: number projects to be opened",
    ),
    terminal_mode: bool = typer.Option(
        False, "--terminal-mode", "-t",
        help="only in list mode: open selected project in terminal.",
    ),
    review: bool = typer.Option(
        False, "--review", "-r",
        help="after showing the status table, enter the semi-automatic commit/revert review "
        "(see 'depman review --help'); reuses this scan instead of rescanning.",
    ),
    order: str = typer.Option(
        "root-last", "--order", click_type=click.Choice(ORDER_CHOICES), help=ORDER_HELP,
    ),
    as_json: bool = typer.Option(
        False, "--json",
        help="print the scan as a single JSON document on stdout instead of the status table "
        "(all diagnostic/progress output goes to stderr). Machine/agent-friendly.",
    ),
    jobs: int = typer.Option(
        8, "--jobs", "-j",
        help="number of repos to fetch/scan concurrently during a live scan (default: 8).",
    ),
):
    """Check for upstream updates in Gitman dependencies (now scans all projects)."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache, jobs=jobs)
    if as_json:
        typer.echo(json.dumps({"configs": loaded_configs, "repos": git_repos}, indent=2, default=str))
        return
    print_check_table(loaded_configs, git_repos, root=root, update_mode=update_mode, list_mode=list_mode, list_open_terminal=terminal_mode)
    if review:
        run_review(root, git_repos, order=order, configs=loaded_configs)


def list_cmd(
    ctx: typer.Context,
    use_cache: bool = typer.Option(
        False, "--use_cache", "-c",
        help="use cached YAML files instead of live scan (path to root).",
    ),
    dirty: bool = typer.Option(False, "--dirty", "-d", help="print only dirty."),
    as_json: bool = typer.Option(
        False, "--json",
        help="print the scan as a single JSON document on stdout instead of the text summary "
        "(all diagnostic/progress output goes to stderr). Machine/agent-friendly.",
    ),
    jobs: int = typer.Option(
        8, "--jobs", "-j",
        help="number of repos to fetch/scan concurrently during a live scan (default: 8).",
    ),
):
    """List all Git/Gitman projects under root."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache, jobs=jobs)
    if as_json:
        typer.echo(json.dumps({"configs": loaded_configs, "repos": git_repos}, indent=2, default=str))
        return
    print_list_configs_repos(loaded_configs, git_repos, only_dirty=dirty)
