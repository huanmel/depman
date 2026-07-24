"""
Update checker using Gitman API + GitPython.
"""

import json

import click

from depman.commands.review import ORDER_CHOICES, ORDER_HELP, run_review
from depman.utils.configs import (
    get_configs_and_repos,
    print_check_table,
    print_list_configs_repos,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
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
@click.option(
    "--review",
    "-r",
    is_flag=True,
    default=False,
    help="after showing the status table, enter the semi-automatic commit/revert review "
    "(see 'depman review --help'); reuses this scan instead of rescanning.",
)
@click.option("--order", type=click.Choice(ORDER_CHOICES), default="root-last", help=ORDER_HELP)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="print the scan as a single JSON document on stdout instead of the status table "
    "(all diagnostic/progress output goes to stderr). Machine/agent-friendly.",
)
@click.option(
    "--jobs",
    "-j",
    type=int,
    default=8,
    help="number of repos to fetch/scan concurrently during a live scan (default: 8).",
)
@click.pass_context
def check_cmd(
    ctx: click.Context,
    use_cache: bool,
    update_mode: bool,
    list_mode: bool,
    terminal_mode: bool,
    review: bool,
    order: str,
    as_json: bool,
    jobs: int,
):
    """Check for upstream updates in Gitman dependencies (now scans all projects)."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache, jobs=jobs)
    if as_json:
        click.echo(json.dumps({"configs": loaded_configs, "repos": git_repos}, indent=2, default=str))
        return
    print_check_table(loaded_configs, git_repos, root=root,update_mode=update_mode,list_mode=list_mode,list_open_terminal=terminal_mode)
    if review:
        run_review(root, git_repos, order=order, configs=loaded_configs)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--use_cache",
    "-c",
    is_flag=True,
    default=False,
    help="use cached YAML files instead of live scan (path to root).",
)
@click.option("--dirty", "-d", is_flag=True, default=False, help="print only dirty.")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="print the scan as a single JSON document on stdout instead of the text summary "
    "(all diagnostic/progress output goes to stderr). Machine/agent-friendly.",
)
@click.option(
    "--jobs",
    "-j",
    type=int,
    default=8,
    help="number of repos to fetch/scan concurrently during a live scan (default: 8).",
)
@click.pass_context
def list_cmd(ctx: click.Context, use_cache: bool, dirty: bool, as_json: bool, jobs: int):
    """List all Git/Gitman projects under root."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache, jobs=jobs)
    if as_json:
        click.echo(json.dumps({"configs": loaded_configs, "repos": git_repos}, indent=2, default=str))
        return
    print_list_configs_repos(loaded_configs, git_repos, only_dirty=dirty)
