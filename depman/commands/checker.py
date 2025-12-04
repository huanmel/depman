"""
Update checker using Gitman API + GitPython.
"""

import click
from pathlib import Path
from git import Repo, GitCommandError
from gitman.models import Config
import yaml  # For manual YAML loads
from rich.console import Console
from rich.table import Table
from depman.utils.configs import  get_configs_and_repos, print_list_configs_repos, print_check_table
from depman import CONFIG_NAME, CACHE_GIT_REPOS, CACHE_CONFIGS


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
# @click.option("--recursive", is_flag=True, help="Flatten deps recursively for updates (legacy).")
# @click.option("--scan-depth", type=int, default=None, help="Max recursion depth for scanning (default: unlimited).")
@click.option("--use-cache",'-c',  is_flag=True, default=False, help="use cached YAML files instead of live scan (path to root).")
@click.pass_context
# , recursive: bool, scan_depth: Optional[int]
def check_cmd(ctx: click.Context, use_cache: bool):
    """Check for upstream updates in Gitman dependencies (now scans all projects)."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache)
    print_check_table(loaded_configs, git_repos,root=root)

    


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--use_cache",'-c',  is_flag=True, default=False, help="use cached YAML files instead of live scan (path to root).")
@click.option("--dirty",'-d',  is_flag=True, default=False, help="print only dirty.")
@click.pass_context
def list_cmd(ctx: click.Context, use_cache: bool, dirty: bool):
    """List all Git/Gitman projects under root."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache)
    print_list_configs_repos(loaded_configs, git_repos, only_dirty=dirty)
    
    


if __name__ == "__main__":
    from depman.cli import cli
    root = r'C:\Users\ivanm\Documents\MATLAB\EKL\deps_tests\gitman_proj_test'
    # cli(['--root', root, 'check-updates'])
    cli(['--root', root, 'list', '--use-cache','--dirty'])
    
