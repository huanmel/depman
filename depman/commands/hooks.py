"""
Hooks subcommand: install/uninstall an optional git pre-push hook that warns
(never blocks) about uncommitted/unpushed changes in a project's gitman-declared
dependencies before a normal `git push`.

The hook is deliberately fast: `hooks check` never fetches over the network,
only inspecting whatever local git state is already on disk, since git blocks
the actual push while a pre-push hook runs.
"""
import stat
import sys
from pathlib import Path
from typing import List, Tuple

import click
from git import Repo

from depman.utils.configs import find_all_configs, gitman_declared_order, local_repo_issues

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

HOOK_MARKER = "# Installed by `depman hooks install`"
HOOK_SCRIPT_TEMPLATE = """#!/bin/sh
{marker} -- warns (never blocks) about uncommitted/unpushed
# changes in this project's gitman-declared dependencies before push.
"{python_exe}" -m depman.cli hooks check
exit 0
"""


@click.group(name="hooks", help="Manage the optional depman pre-push warning hook.")
@click.pass_context
def hooks(ctx: click.Context):
    pass


@hooks.command("install")
@click.option("--force", is_flag=True, help="overwrite an existing pre-push hook.")
@click.pass_context
def hooks_install(ctx: click.Context, force: bool):
    """Install the pre-push warning hook into this repo (the resolved --root)."""
    root: Path = ctx.obj["root"]
    hook_path = root / ".git" / "hooks" / "pre-push"
    if hook_path.exists() and not force:
        if HOOK_MARKER in hook_path.read_text(errors="ignore"):
            click.echo(click.style(
                f"depman's pre-push hook is already installed at {hook_path}. "
                "Use --force to reinstall.", fg="yellow"))
        else:
            click.echo(click.style(
                f"❌ {hook_path} already exists and wasn't installed by depman "
                "-- use --force to overwrite it.", fg="red"))
        raise SystemExit(1)

    script = HOOK_SCRIPT_TEMPLATE.format(marker=HOOK_MARKER, python_exe=sys.executable)
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(script)
    mode = hook_path.stat().st_mode
    hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    click.echo(click.style(f"✅ Installed pre-push hook at {hook_path}", fg="green"))


@hooks.command("uninstall")
@click.pass_context
def hooks_uninstall(ctx: click.Context):
    """Remove the pre-push hook, but only if depman installed it."""
    root: Path = ctx.obj["root"]
    hook_path = root / ".git" / "hooks" / "pre-push"
    if not hook_path.exists():
        click.echo("No pre-push hook installed.")
        return
    if HOOK_MARKER not in hook_path.read_text(errors="ignore"):
        click.echo(click.style(
            f"❌ {hook_path} wasn't installed by depman -- leaving it alone.", fg="red"))
        return
    hook_path.unlink()
    click.echo(click.style(f"✅ Removed pre-push hook at {hook_path}", fg="green"))


def _collect_issues(root: Path) -> List[Tuple[str, dict]]:
    configs, _ = find_all_configs(root, {"repos": {}})  # no fetch: pure YAML parsing
    issues = []
    for repo_path in gitman_declared_order(configs):
        repo_dir = root / repo_path
        if not (repo_dir / ".git").exists():
            continue
        try:
            repo = Repo(repo_dir)
        except Exception:
            continue
        info = local_repo_issues(repo)
        if info["has_uncommitted"] or info["has_unpushed"]:
            issues.append((repo_path, info))
    return issues


@hooks.command("check")
@click.pass_context
def hooks_check(ctx: click.Context):
    """
    Fast, local-only (no fetch) check of this project's gitman-declared deps for
    uncommitted/unpushed issues. Always exits 0 -- warn-only, meant to be called
    from the installed pre-push hook, or run directly to preview what it says.
    """
    root: Path = ctx.obj["root"]
    issues = _collect_issues(root)
    if not issues:
        return
    # Plain ASCII only here (no emoji): this runs unattended from a pre-push hook
    # in whatever console codepage the user's shell happens to have (commonly
    # cp1252 on Windows, not UTF-8), and a UnicodeEncodeError here would print a
    # scary traceback on every push even though the hook's own `exit 0` still
    # keeps the push itself from being blocked.
    click.echo(click.style(f"[depman] {len(issues)} repo(s) have local issues:", fg="yellow"))
    for repo_path, info in issues:
        parts = []
        if info["has_uncommitted"]:
            parts.append(f"{len(info['uncommitted_files'])} uncommitted file(s)")
        if info["has_unpushed"]:
            parts.append(f"{info['unpushed_count']} unpushed commit(s)")
        click.echo(click.style(f"   {repo_path}: {', '.join(parts)}", fg="yellow"))
