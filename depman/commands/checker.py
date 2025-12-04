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
from depman.utils.configs import  get_configs_and_repos, print_list_configs_repos
from depman import CONFIG_NAME, CACHE_GIT_REPOS, CACHE_CONFIGS


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
# @click.option("--recursive", is_flag=True, help="Flatten deps recursively for updates (legacy).")
# @click.option("--scan-depth", type=int, default=None, help="Max recursion depth for scanning (default: unlimited).")
@click.option("--use-cache",  is_flag=True, default=False, help="use cached YAML files instead of live scan (path to root).")
@click.pass_context
# , recursive: bool, scan_depth: Optional[int]
def check_updates_cmd(ctx: click.Context, use_cache: bool):
    """Check for upstream updates in Gitman dependencies (now scans all projects)."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache)

    # Table for updates
    table = Table(title="Git deps status", show_header=True,
                  header_style="bold magenta")
    cols = ["Dep", "Status", "Uncommitted",
            "Unpushed", "Update", "Update Main"]
    for col in cols:
        table.add_column(col, style="dim", overflow="fold")

    has_updates = False
    click.echo(click.style(
        f"\n=== Git Repos Status (scanned at {git_repos['datetime']}) ===", bold=True))

    for key, val in git_repos["repos"].items():
        has_updates_repo = val["has_uncommitted"] or val["has_unpushed"] or val["has_update"] or val["has_update_main"]

        if val["has_uncommitted"]:
            click.echo(click.style(
                f"⚠️  Repo {key} has uncommitted changes: {val['uncommitted_files']}", fg="yellow"))
            has_updates_repo = True
        if val["has_unpushed"]:
            click.echo(click.style(
                f"⚠️  Repo {key} has {val['unpushed_count']} unpushed commits", fg="yellow"))
        if val["has_update"]:
            details = val["update_details"]
            click.echo(click.style(
                f"⚠️  Repo {key} has updates on branch {details['branch']}: {details['latest_hash'][:7]} - {details['message']}", fg="yellow"))
        if val["has_update_main"]:
            details = val["update_details_main"]
            click.echo(click.style(
                f"⚠️  Repo {key} has updates on main branch {details['branch']}: {details['latest_hash'][:7]} - {details['message']}", fg="yellow"))

        has_updates = has_updates or has_updates_repo
        table.add_row(*[
            key,
            "⚠️" if has_updates_repo else "✅",
            "⚠️" if val["has_uncommitted"] else "✅",
            "⚠️" if val["has_unpushed"] else "✅",
            "⚠️" if val["has_update"] else "✅",
            "⚠️" if val["has_update_main"] else "✅"
        ],  style='bright_green' if not has_updates_repo else 'bright_yellow')
        # table.add_row(
        #     key,
        #     "!" if has_updates_repo else "ok",
        #     "!" if val["has_uncommitted"] else "ok",
        #     "!" if val["has_unpushed"] else "ok",
        #     "!" if val["has_update"] else "ok",
        #     "!" if val["has_update_main"] else "ok",
        #     style='bright_green' if not has_updates_repo else 'bright_yellow'
        # )

    console = Console()
    console.print(table)
    if has_updates:
        click.echo(click.style(
            "⚠️  Updates available—run 'depman gm update' to apply.", fg="yellow"))
    return  # Early return to avoid full project scan for now
    for project_root, config in projects:
        click.echo(click.style(
            f"\n=== Project: {project_root} ===", bold=True))
        print_project_tree(config, project_root)

        location = project_root / \
            config.content.get("location", "requirements")
        all_reqs = get_all_requirements(config, recursive)

        if not all_reqs:
            click.echo("No requirements found.")
            continue

        # Table for updates
        table = click.table(
            [("Dep", "Status", "Repo")],
            [("", "", "")],  # Header row
            headers=False,
            colalign=("left", "left", "left"),
        )
        table.add_row(["", "", ""])  # Spacer

        has_updates = False
        for req in all_reqs:
            name = req.get("name", Path(req["repo"]).name)
            repo_url = req["repo"]
            target_rev = req["rev"]
            dep_path = location / name

            if not dep_path.exists():
                status = click.style("Not installed", fg="yellow")
                table.add_row([name, status, repo_url])
                continue

            try:
                repo = Repo(dep_path)
                repo.remotes.origin.fetch()

                current_sha = repo.head.commit.hexsha
                try:
                    origin_ref = repo.refs[f"origin/{target_rev}"]
                    latest_sha = origin_ref.commit.hexsha
                    is_branch = True
                except IndexError:
                    latest_sha = repo.rev_parse(target_rev).hexsha
                    is_branch = False

                short_current = current_sha[:8]
                short_latest = latest_sha[:8]

                if current_sha == latest_sha:
                    status = click.style("Up to date", fg="green")
                else:
                    has_updates = True
                    if is_branch:
                        status = click.style(
                            f"Update avail. ({short_current} → {short_latest})", fg="red")
                    else:
                        status = click.style(
                            f"Mismatch ({short_latest})", fg="yellow")

                table.add_row([name, status, repo_url])

            except GitCommandError as e:
                status = click.style(f"Git error ({e})", fg="red")
                table.add_row([name, status, repo_url])
            except KeyError as e:
                status = click.style(f"Missing key ({e})", fg="yellow")
                table.add_row([name, status, repo_url])

        click.echo(table)
        if has_updates:
            click.echo(click.style(
                "⚠️  Updates available—run 'depman gm update' to apply.", fg="yellow"))


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--use-cache",  is_flag=True, default=False, help="use cached YAML files instead of live scan (path to root).")
@click.option("--dirty",  is_flag=True, default=False, help="print only dirty.")
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
    
