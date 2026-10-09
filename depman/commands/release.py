"""
Release subcommand: lock -> version/custom scripts -> commit -> annotated tag -> push.

The version is resolved up front -- either given explicitly (v0.1.2) or bumped
(major/minor/fix) from the highest semver tag reachable from HEAD -- and shown
together with the whole plan for confirmation before anything is changed.
Scripts come from `--version-script` / `--pre-commit-script` flags, falling back
to a `release:` section in `.depman.yml`. Commands are run through the shell in
the repo root with `{version}` / `{tag}` substituted and DEPMAN_VERSION /
DEPMAN_TAG exported. Tags are annotated because `git push --follow-tags` only
pushes annotated tags.
"""
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import yaml
from git import GitCommandError, Repo

from depman import CONFIG_NAME, DEPMAN_CONFIG_NAME

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

BUMP_KINDS = ("major", "minor", "fix", "patch")
_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

Version = Tuple[int, int, int]


def parse_version(text: str) -> Optional[Version]:
    """'v1.2.3' / '1.2.3' -> (1, 2, 3); None if not a plain semver."""
    match = _VERSION_RE.match(text.strip())
    return tuple(int(g) for g in match.groups()) if match else None  # type: ignore[return-value]


def format_tag(version: Version) -> str:
    return "v{}.{}.{}".format(*version)


def latest_version(repo: Repo) -> Optional[Version]:
    """Highest semver tag reachable from HEAD, or None if there is none."""
    output = repo.git.tag("--merged", "HEAD")
    versions = [v for v in (parse_version(t) for t in output.splitlines()) if v]
    return max(versions) if versions else None


def resolve_version(target: str, previous: Optional[Version]) -> Version:
    """target is major|minor|fix|patch (bump `previous`, v0.0.0 if None) or an explicit version."""
    explicit = parse_version(target)
    if explicit:
        return explicit
    if target not in BUMP_KINDS:
        raise ValueError(
            f"'{target}' is neither major/minor/fix nor a version like v0.1.2")
    major, minor, patch = previous or (0, 0, 0)
    if target == "major":
        return (major + 1, 0, 0)
    if target == "minor":
        return (major, minor + 1, 0)
    return (major, minor, patch + 1)


def load_release_config(root: Path) -> Dict[str, Any]:
    path = root / DEPMAN_CONFIG_NAME
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("release") or {}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _run_script(command: str, root: Path, version: Version) -> None:
    tag = format_tag(version)
    version_str = tag[1:]
    expanded = command.replace("{version}", version_str).replace("{tag}", tag)
    env = dict(os.environ, DEPMAN_VERSION=version_str, DEPMAN_TAG=tag)
    typer.echo(typer.style(f"$ {expanded}", dim=True))
    result = subprocess.run(expanded, shell=True, cwd=root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"script failed (exit {result.returncode}): {expanded}")


def _changed_files(repo: Repo) -> List[str]:
    return [line for line in repo.git.status("--porcelain").splitlines() if line]


def _abort(message: str) -> typer.Exit:
    typer.echo(typer.style(f"❌ {message}", fg="red"))
    return typer.Exit(1)


def _confirm(prompt: str, default: bool = True) -> None:
    if not typer.confirm(prompt, default=default):
        typer.echo("Cancelled.")
        raise typer.Exit(1)


def release_cmd(
    ctx: typer.Context,
    target: str = typer.Argument(
        ..., help="major | minor | fix (bump the latest version tag) or an explicit version like v0.1.2."),
    version_script: Optional[List[str]] = typer.Option(
        None, "--version-script", help="command that updates the version file(s); repeatable. "
        "{version}/{tag} are substituted. Overrides release.version_scripts in .depman.yml."),
    pre_commit_script: Optional[List[str]] = typer.Option(
        None, "--pre-commit-script", help="extra command to run before the commit; repeatable. "
        "Overrides release.pre_commit_scripts in .depman.yml."),
    no_lock: bool = typer.Option(False, "--no-lock", help="skip `gitman lock`."),
    force_tag: bool = typer.Option(
        False, "--force-tag", help="allow moving an existing tag to the new commit "
        "(rewrites it locally, then force-pushes that tag to the remote)."),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="commit message (default: 'Release <tag>')."),
    remote: Optional[str] = typer.Option(None, "--remote", help="remote to push to (default: origin)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="resolve and print the plan, change nothing."),
):
    """Build a release: lock, bump version, run scripts, commit, tag, push."""
    root: Path = ctx.obj["root"]
    config = load_release_config(root)
    version_scripts = list(version_script) if version_script else _as_list(config.get("version_scripts"))
    pre_commit_scripts = list(pre_commit_script) if pre_commit_script else _as_list(config.get("pre_commit_scripts"))
    remote_name = remote or config.get("remote", "origin")
    do_lock = not no_lock and config.get("lock", True) and (root / CONFIG_NAME).is_file()

    repo = Repo(root)
    if repo.head.is_detached:
        raise _abort("Detached HEAD; check out a branch before releasing.")
    if remote_name not in [r.name for r in repo.remotes]:
        raise _abort(f"No '{remote_name}' remote configured.")
    branch = repo.active_branch.name

    previous = latest_version(repo)
    try:
        version = resolve_version(target, previous)
    except ValueError as e:
        raise _abort(str(e))
    tag = format_tag(version)
    tag_exists = tag in [t.name for t in repo.tags]
    if tag_exists and not force_tag:
        raise _abort(f"Tag {tag} already exists. Use --force-tag to move it to the new commit.")

    pending = _changed_files(repo)
    typer.echo(typer.style("\n=== Release plan ===", bold=True, fg="cyan"))
    typer.echo(f"  previous version : {format_tag(previous) if previous else '(none)'}")
    typer.echo(f"  new version      : {typer.style(tag, bold=True, fg='green')}")
    typer.echo(f"  branch / remote  : {branch} -> {remote_name}")
    typer.echo(f"  gitman lock      : {'yes' if do_lock else 'no'}")
    for cmd in version_scripts:
        typer.echo(f"  version script   : {cmd}")
    for cmd in pre_commit_scripts:
        typer.echo(f"  pre-commit script: {cmd}")
    typer.echo(f"  push             : git push {remote_name} {branch} --follow-tags")
    if tag_exists:
        typer.echo(typer.style(
            f"  ⚠️  tag {tag} EXISTS at {repo.tags[tag].commit.hexsha[:7]} and will be rewritten "
            f"locally, then force-pushed to {remote_name}.", fg="yellow"))
    if pending:
        typer.echo(typer.style(
            f"  ⚠️  {len(pending)} uncommitted change(s) already present; they will be included "
            "in the release commit.", fg="yellow"))
    if dry_run:
        typer.echo("\nDry run: nothing changed.")
        return
    _confirm(f"\nProceed with release {tag}?")

    try:
        if do_lock:
            from gitman import lock
            typer.echo(typer.style("Running gitman lock...", fg="cyan"))
            lock(root=root)
        for cmd in version_scripts + pre_commit_scripts:
            _run_script(cmd, root, version)
    except Exception as e:
        raise _abort(f"{e}\nNothing was committed; any changes made so far remain in the working tree.")

    files = _changed_files(repo)
    if files:
        typer.echo(typer.style("\nAbout to commit:", fg="green"))
        for line in files:
            typer.echo(f"  {line}")
        commit_message = typer.prompt("Commit message", default=message or f"Release {tag}")
        _confirm(f"Commit all {len(files)} file(s) and tag {tag}?")
        try:
            repo.git.add(A=True)
            repo.git.commit("-m", commit_message)
        except GitCommandError as e:
            raise _abort(f"Commit failed (hook?): {e}")
        typer.echo(typer.style(f"✅ Committed: {commit_message}", fg="green"))
    else:
        typer.echo(typer.style(
            f"\nNo changes to commit; tagging current HEAD ({repo.head.commit.hexsha[:7]}).", fg="yellow"))
        _confirm(f"Tag HEAD as {tag}?")

    tag_args = ["-a", "-f", tag] if tag_exists else ["-a", tag]
    try:
        repo.git.tag(*tag_args, "-m", f"Release {tag}")
    except GitCommandError as e:
        raise _abort(f"Tagging failed: {e}")
    typer.echo(typer.style(f"✅ Tagged {tag} at {repo.head.commit.hexsha[:7]}", fg="green"))

    push_desc = f"git push {remote_name} {branch} --follow-tags"
    if tag_exists:
        push_desc += f"  +  git push --force {remote_name} refs/tags/{tag}"
    _confirm(f"Push now? ({push_desc})", default=False)
    try:
        repo.git.push(remote_name, branch, "--follow-tags")
        if tag_exists:
            repo.git.push("--force", remote_name, f"refs/tags/{tag}")
    except GitCommandError as e:
        raise _abort(f"Push failed: {e}\nCommit and tag exist locally; fix the problem and push manually.")
    typer.echo(typer.style(f"🚀 Released {tag}", fg="green", bold=True))
