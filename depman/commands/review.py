"""
Review subcommand: semi-automatic per-repo commit/revert workflow.

Walks every repo with uncommitted changes (root and nested deps), shows a
compact list of changed files, and asks for a per-repo decision. Full diffs
are shown on demand ("d") rather than dumped up front, since large repos can
generate a lot of diff output. "Revert" never hard-deletes: it stashes the
changes (including untracked files) under a labeled stash so they can always
be recovered with `git stash pop`. `run_review` is shared with `depman check
-r`, which reuses the scan `check` already did instead of rescanning.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
from git import GitCommandError, Repo

from depman import CACHE_CONFIGS, CACHE_GIT_REPOS
from depman.utils.configs import find_all_git_repos, get_cached_configs

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

ORDER_CHOICES = ["root-last", "root-first"]
ORDER_HELP = (
    "processing order for dirty repos: 'root-last' (default) handles nested deps "
    "first and the root project last; 'root-first' goes alphabetically (root first)."
)


def _changed_files(repo: Repo) -> List[Tuple[str, str]]:
    """Parse `git status --porcelain` into a list of (status_code, path)."""
    output = repo.git.status("--porcelain")
    files = []
    for line in output.splitlines():
        if not line:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:  # renames: "old -> new"
            path = path.split(" -> ", 1)[1]
        files.append((code, path))
    return files


def _print_file_list(files: List[Tuple[str, str]]) -> None:
    click.echo(click.style("Changed files:", fg="yellow"))
    for i, (code, path) in enumerate(files):
        click.echo(f"  [{i}] {code.strip() or '??':<3} {path}")


def _diff_for_file(repo: Repo, path: str) -> str:
    staged = repo.git.diff("--cached", "--", path)
    unstaged = repo.git.diff("--", path)
    if not staged and not unstaged:
        full_path = Path(repo.working_tree_dir) / path
        if full_path.is_file():
            try:
                return "(untracked file)\n" + full_path.read_text(errors="replace")
            except UnicodeDecodeError:
                return "(untracked binary file)"
        return "(no diff available)"
    parts = []
    if staged:
        parts.append("--- staged ---\n" + staged)
    if unstaged:
        parts.append("--- unstaged ---\n" + unstaged)
    return "\n".join(parts)


def _show_diff_menu(repo: Repo, files: List[Tuple[str, str]]) -> None:
    if not files:
        return
    selection = click.prompt(
        "Show diff for file # (blank = all)", default="", show_default=False
    )
    if selection.strip() == "":
        targets = [path for _, path in files]
    elif selection.isdigit() and 0 <= int(selection) < len(files):
        targets = [files[int(selection)][1]]
    else:
        click.echo(click.style(f"Invalid selection: {selection}", fg="red"))
        return
    for path in targets:
        click.echo(click.style(f"--- {path} ---", dim=True))
        click.echo(_diff_for_file(repo, path))


def _commit_repo(repo: Repo, repo_path: str, files: List[Tuple[str, str]]) -> None:
    new_files = [path for code, path in files if code.strip() in ("??", "A")]
    click.echo(click.style("About to commit:", fg="green"))
    _print_file_list(files)
    if new_files:
        click.echo(click.style(
            f"  ({len(new_files)} new/untracked file(s) will be added)", fg="yellow"))
    if not click.confirm(f"Commit all {len(files)} file(s) in {repo_path}?", default=True):
        click.echo("Cancelled.")
        return
    message = click.prompt("Commit message", default=f"wip: {repo_path}")
    repo.git.add(A=True)
    repo.index.commit(message)
    click.echo(click.style(f"✅ Committed {repo_path}: {message}", fg="green"))


def _revert_repo(repo: Repo, repo_path: str) -> None:
    stash_name = f"depman-revert-safety-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        repo.git.stash("push", "-u", "-m", stash_name)
    except GitCommandError as e:
        click.echo(click.style(f"❌ Could not revert {repo_path}: {e}", fg="red"))
        return
    click.echo(click.style(
        f"⚠️  Reverted {repo_path} — changes safely stashed as '{stash_name}'.", fg="yellow"))
    click.echo(f'    Recover with: git -C "{repo.working_tree_dir}" stash pop')


def _review_repo(repo: Repo, repo_path: str) -> Optional[str]:
    """Interact with the user about a single dirty repo. Returns 'quit' to stop the whole review."""
    while True:
        click.echo(click.style(f"\n=== {repo_path} ===", bold=True, fg="cyan"))
        files = _changed_files(repo)
        _print_file_list(files)

        choice = click.prompt(
            "commit / revert / diff / skip / quit",
            type=click.Choice(["c", "r", "d", "s", "q"]),
            default="s",
            show_choices=True,
        )

        if choice == "q":
            return "quit"
        elif choice == "c":
            _commit_repo(repo, repo_path, files)
            return None
        elif choice == "r":
            if click.confirm(
                f"Really revert {repo_path}? Changes will be stashed (recoverable) "
                "and the working tree reset to HEAD",
                default=False,
            ):
                _revert_repo(repo, repo_path)
            else:
                click.echo("Cancelled.")
            return None
        elif choice == "d":
            _show_diff_menu(repo, files)
            # loop back: re-show the (still up to date) file list and re-prompt
        else:
            click.echo(f"Skipped {repo_path}")
            return None


def _ordered_dirty_paths(dirty_paths: Iterable[str], order: str) -> List[str]:
    paths = sorted(dirty_paths)
    if order == "root-last" and "." in paths:
        paths.remove(".")
        paths.append(".")
    return paths


def run_review(root: Path, git_repos: Dict[str, Any], order: str = "root-last") -> None:
    """Shared review workflow, usable from `depman review` and `depman check -r`."""
    dirty = {
        path: info for path, info in git_repos["repos"].items() if info["has_uncommitted"]
    }
    if not dirty:
        click.echo(click.style("✅ No uncommitted changes in any repo.", fg="green"))
        return

    click.echo(click.style(f"\nFound {len(dirty)} repo(s) with uncommitted changes.", bold=True))

    for repo_path in _ordered_dirty_paths(dirty.keys(), order):
        repo = Repo(root / repo_path)
        if _review_repo(repo, repo_path) == "quit":
            click.echo("Stopping review.")
            break

    click.echo(click.style("\nReview complete.", bold=True))


@click.command("review", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--use-cache",
    "-c",
    is_flag=True,
    default=False,
    help="use the YAML snapshot from a previous 'check'/'list' scan instead of rescanning "
    "(faster, no network fetch; which repos are dirty may be stale if scanned a while ago).",
)
@click.option("--order", type=click.Choice(ORDER_CHOICES), default="root-last", help=ORDER_HELP)
@click.pass_context
def review_cmd(ctx: click.Context, use_cache: bool, order: str):
    """Semi-automatic review: commit or revert changes in each dirty repo, one at a time."""
    root: Path = ctx.obj["root"]
    if use_cache:
        _, git_repos = get_cached_configs(root, CACHE_GIT_REPOS, CACHE_CONFIGS)
    else:
        git_repos = find_all_git_repos(root)
    run_review(root, git_repos, order=order)
