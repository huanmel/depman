"""
Review subcommand: semi-automatic per-repo commit/revert/push workflow.

Walks every repo that needs attention (root and nested deps) -- either it has
uncommitted changes, or it's clean but has commits not yet pushed to origin --
shows a compact list of changed files, and asks for a per-repo decision. Full
diffs are shown on demand ("d") rather than dumped up front, since large
repos can generate a lot of diff output. "Revert" never hard-deletes: it
stashes the changes (including untracked files) under a labeled stash so
they can always be recovered with `git stash pop`. `run_review` is shared
with `depman check -r`, which reuses the scan `check` already did instead of
rescanning.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import typer
from git import GitCommandError, Repo
from git.exc import HookExecutionError

from depman.backends import get_backend
from depman.utils.configs import get_configs_and_repos, gitman_declared_order

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

ORDER_CHOICES = ["root-last", "root-first", "gitman"]
ORDER_HELP = (
    "processing order for dirty repos: 'root-last' (default) handles nested deps "
    "alphabetically, root last; 'root-first' is the same but root first; 'gitman' "
    "follows each gitman.yml's declared dep order (depth-first through nested "
    "configs), root last -- reorder gitman.yml's sources to control this order."
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
    typer.echo(typer.style("Changed files:", fg="yellow"))
    for i, (code, path) in enumerate(files):
        typer.echo(f"  [{i}] {code.strip() or '??':<3} {path}")


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
    selection = typer.prompt(
        "Show diff for file # (blank = all)", default="", show_default=False
    )
    if selection.strip() == "":
        targets = [path for _, path in files]
    elif selection.isdigit() and 0 <= int(selection) < len(files):
        targets = [files[int(selection)][1]]
    else:
        typer.echo(typer.style(f"Invalid selection: {selection}", fg="red"))
        return
    for path in targets:
        typer.echo(typer.style(f"--- {path} ---", dim=True))
        typer.echo(_diff_for_file(repo, path))


def _consuming_configs_needing_relock(
    configs: Optional[Dict[str, Any]], repo_path: str, new_rev: str
) -> List[str]:
    """
    Which OTHER configs declare repo_path as a dependency with a real lock (rev_locked
    is not None) that new_rev no longer matches? This is the mirror image of
    _stale_locked_deps: that looks at a config's own children, this looks at who
    consumes repo_path as a child. A dep with no lock at all is deliberately excluded,
    same rationale as _stale_locked_deps.
    """
    if not configs:
        return []
    consumers = []
    for config_path, config_entry in configs.get("configs", {}).items():
        dep_info = config_entry.get("deps", {}).get(repo_path)
        if not dep_info:
            continue
        rev_locked = dep_info.get("rev_locked")
        if rev_locked is not None and rev_locked != new_rev:
            consumers.append(config_path)
    return consumers


def _warn_consuming_configs_stale(
    repo: Repo, repo_path: str, configs: Optional[Dict[str, Any]]
) -> None:
    new_rev = repo.head.commit.hexsha
    consumers = _consuming_configs_needing_relock(configs, repo_path, new_rev)
    if not consumers:
        return
    typer.echo(typer.style(
        f"⚠️  This commit leaves {repo_path} out of sync with the lock in "
        f"{len(consumers)} config(s): {', '.join(consumers)} -- consider relocking "
        "there before pushing.", fg="yellow"))


def _commit_repo(
    repo: Repo, repo_path: str, files: List[Tuple[str, str]],
    configs: Optional[Dict[str, Any]] = None,
) -> bool:
    """Returns True if a commit was made, False if the user cancelled."""
    _offer_lock_before_commit(repo, repo_path, configs)
    files = _changed_files(repo)  # refresh: a lock just run above may have changed gitman.yml
    new_files = [path for code, path in files if code.strip() in ("??", "A")]
    typer.echo(typer.style("About to commit:", fg="green"))
    _print_file_list(files)
    if new_files:
        typer.echo(typer.style(
            f"  ({len(new_files)} new/untracked file(s) will be added)", fg="yellow"))
    if not typer.confirm(f"Commit all {len(files)} file(s) in {repo_path}?", default=True):
        typer.echo("Cancelled.")
        return False
    message = typer.prompt("Commit message", default=f"wip: {repo_path}")
    repo.git.add(A=True)
    try:
        repo.index.commit(message)
    except HookExecutionError as e:
        # The commit itself already succeeded -- post-commit hooks run after the
        # commit object and branch ref exist, and per githooks(5) are advisory
        # only ("cannot affect the outcome of git commit"). GitPython raises
        # anyway on a non-zero hook exit. Common cause: the repo is configured
        # for git-lfs (has a git-lfs post-commit hook) but 'git-lfs' isn't on
        # PATH in this environment.
        typer.echo(typer.style(
            f"⚠️  Committed {repo_path}: {message} (but its post-commit hook failed "
            f"-- commit succeeded regardless): {e}", fg="yellow"))
        _warn_consuming_configs_stale(repo, repo_path, configs)
        return True
    typer.echo(typer.style(f"✅ Committed {repo_path}: {message}", fg="green"))
    _warn_consuming_configs_stale(repo, repo_path, configs)
    return True


def _unpushed_commits(repo: Repo, branch: str) -> List:
    """Commits on `branch` not yet on `origin/branch` (best-effort fetch first for accuracy)."""
    try:
        repo.remotes.origin.fetch()
    except GitCommandError:
        pass  # offline/unreachable; fall back to whatever we already know locally

    remote_ref = f"origin/{branch}"
    try:
        repo.refs[remote_ref]
    except IndexError:
        return list(repo.iter_commits(branch))  # branch was never pushed before
    return list(repo.iter_commits(f"{remote_ref}..{branch}"))


def _print_unpushed_commits(commits: List) -> None:
    if not commits:
        typer.echo(typer.style("  (no new commits to push)", fg="yellow"))
        return
    typer.echo(typer.style(f"Commits to push ({len(commits)}):", fg="yellow"))
    for commit in commits:
        typer.echo(f"  {commit.hexsha[:7]}  {commit.message.splitlines()[0]}")


def _squash_commit_message(commits: List) -> str:
    """Combine each squashed commit's full message, oldest first, into one multiline message."""
    return "\n\n".join(commit.message.strip() for commit in reversed(commits))


def _squash_unpushed_commits(repo: Repo, repo_path: str, commits: List) -> bool:
    """
    Offer to squash `commits` -- exactly origin/<branch>..<branch>, i.e. commits not yet
    on the remote -- into a single commit. Always safe to push normally afterward, since
    none of these commits exist on origin yet (no rewriting of shared history). Returns
    True if a squash happened.
    """
    if len(commits) < 2:
        return False
    oldest = commits[-1]
    if not oldest.parents:
        typer.echo(typer.style(
            "Cannot squash: that would reach the repo's very first commit.", fg="red"))
        return False

    default_message = _squash_commit_message(commits)
    typer.echo(typer.style(f"Combine these {len(commits)} commits into one:", fg="green"))
    typer.echo(default_message)
    if not typer.confirm(f"Squash {len(commits)} commits into one before pushing?", default=False):
        return False

    message = typer.prompt(
        "Commit message (blank = use combined message above)", default="", show_default=False
    )
    final_message = message.strip() or default_message
    repo.git.reset("--soft", oldest.parents[0].hexsha)
    repo.git.commit("-m", final_message)
    typer.echo(typer.style(f"✅ Squashed {len(commits)} commits into one in {repo_path}.", fg="green"))
    return True


def _push_repo(repo: Repo, repo_path: str) -> None:
    if not hasattr(repo.remotes, "origin"):
        typer.echo(typer.style(f"❌ {repo_path} has no 'origin' remote configured.", fg="red"))
        return
    if repo.head.is_detached:
        typer.echo(typer.style(
            f"❌ {repo_path} is in detached HEAD state; check out a branch before pushing.",
            fg="red"))
        return
    branch = repo.active_branch.name
    commits = _unpushed_commits(repo, branch)
    _print_unpushed_commits(commits)
    if not commits:
        return

    if len(commits) > 1 and _squash_unpushed_commits(repo, repo_path, commits):
        commits = _unpushed_commits(repo, branch)
        _print_unpushed_commits(commits)

    if not typer.confirm(f"Push {repo_path} ({branch}) to origin?", default=False):
        typer.echo("Cancelled.")
        return
    try:
        output = repo.git.push("origin", branch)
    except GitCommandError as e:
        typer.echo(typer.style(f"❌ Push failed for {repo_path}: {e}", fg="red"))
        return
    typer.echo(typer.style(f"✅ Pushed {repo_path} ({branch}) to origin.", fg="green"))
    if output:
        typer.echo(output)


def _revert_repo(repo: Repo, repo_path: str) -> None:
    stash_name = f"depman-revert-safety-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        repo.git.stash("push", "-u", "-m", stash_name)
    except GitCommandError as e:
        typer.echo(typer.style(f"❌ Could not revert {repo_path}: {e}", fg="red"))
        return
    typer.echo(typer.style(
        f"⚠️  Reverted {repo_path} — changes safely stashed as '{stash_name}'.", fg="yellow"))
    typer.echo(f'    Recover with: git -C "{repo.working_tree_dir}" stash pop')


def _stale_locked_deps(configs: Optional[Dict[str, Any]], repo_path: str) -> List[str]:
    """
    Dep names in repo_path's OWN gitman.yml that ARE locked (have a rev_locked) but
    whose locked rev doesn't match what's actually installed. A dep with no lock at
    all (rev_locked is None -- the project simply doesn't use gitman's locking) is
    not "stale", it's just unlocked, so it's excluded here.
    """
    if not configs:
        return []
    config_entry = configs.get("configs", {}).get(repo_path)
    if not config_entry:
        return []
    return [
        dep_info.get("name", dep_path)
        for dep_path, dep_info in config_entry.get("deps", {}).items()
        if dep_info.get("rev_locked") is not None and dep_info.get("rev_match") is False
    ]


def _relock_config(
    repo: Repo, repo_path: str, dep_names: List[str], configs: Optional[Dict[str, Any]]
) -> bool:
    """
    Runs the config backend's relock() for repo_path's own config. gitman's lock()
    relocks every declared dep to its currently-installed revision in one shot (it
    doesn't support locking a subset), so dep_names is passed through for the
    interface but not relied on here. Returns True on success and refreshes the
    in-memory configs snapshot to match -- otherwise the same staleness/unlocked-ness
    would keep getting reported every loop iteration, since configs was only scanned
    once before this review started.
    """
    backend_name = (configs or {}).get("configs", {}).get(repo_path, {}).get("backend")
    backend = get_backend(backend_name) if backend_name else None
    if backend is None:
        typer.echo(typer.style(
            f"❌ No config backend registered for '{backend_name}' -- can't relock {repo_path}.",
            fg="red"))
        return False
    try:
        backend.relock(Path(repo.working_tree_dir), dep_names)
    except Exception as e:
        typer.echo(typer.style(f"❌ Relock failed for {repo_path}: {e}", fg="red"))
        return False
    typer.echo(typer.style(f"✅ Relocked gitman.yml in {repo_path}.", fg="green"))
    if configs:
        for dep_info in configs.get("configs", {}).get(repo_path, {}).get("deps", {}).values():
            dep_info["rev_match"] = True
            dep_info["rev_locked"] = dep_info.get("rev_installed")
    return True


def _offer_manual_relock(
    repo: Repo, repo_path: str, configs: Optional[Dict[str, Any]]
) -> None:
    """Manual counterpart to the top-level 'u' menu action: offered whenever repo_path
    owns a config file at all, not just when something is actually unlocked/stale."""
    summary = _describe_pending_lock(configs, repo_path)
    if summary is not None:
        prompt = (
            f"Relock gitman.yml in {repo_path} for {summary}? This writes the "
            "currently-installed revisions into sources_locked."
        )
    else:
        prompt = (
            f"{repo_path}'s gitman.yml has no unlocked or stale deps right now. "
            "Run 'gitman lock' anyway?"
        )
    if not typer.confirm(prompt, default=False):
        typer.echo("Cancelled.")
        return
    unlocked = _unlocked_deps(configs, repo_path)
    stale = _stale_locked_deps(configs, repo_path)
    _relock_config(repo, repo_path, unlocked + stale, configs)


def _unlocked_deps(configs: Optional[Dict[str, Any]], repo_path: str) -> List[str]:
    """
    Dep names in repo_path's OWN gitman.yml that have no lock at all (rev_locked is
    None) -- the mirror image of _stale_locked_deps's guard, which deliberately
    excludes these as "just unlocked, not stale".
    """
    if not configs:
        return []
    config_entry = configs.get("configs", {}).get(repo_path)
    if not config_entry:
        return []
    return [
        dep_info.get("name", dep_path)
        for dep_path, dep_info in config_entry.get("deps", {}).items()
        if dep_info.get("rev_locked") is None
    ]


def _has_config(configs: Optional[Dict[str, Any]], repo_path: str) -> bool:
    """Whether repo_path owns a config file of its own (e.g. a gitman.yml), regardless
    of how many deps it declares or their lock state."""
    return bool((configs or {}).get("configs", {}).get(repo_path))


def _describe_pending_lock(configs: Optional[Dict[str, Any]], repo_path: str) -> Optional[str]:
    """Human-readable 'N unlocked (...); N stale (...)' summary of repo_path's own
    deps needing a lock, or None if it has no config or nothing is pending."""
    unlocked = _unlocked_deps(configs, repo_path)
    stale = _stale_locked_deps(configs, repo_path)
    if not unlocked and not stale:
        return None
    parts = []
    if unlocked:
        parts.append(f"{len(unlocked)} unlocked ({', '.join(unlocked)})")
    if stale:
        parts.append(f"{len(stale)} stale ({', '.join(stale)})")
    return "; ".join(parts)


def _offer_lock_before_commit(
    repo: Repo, repo_path: str, configs: Optional[Dict[str, Any]]
) -> None:
    """
    Before committing, offer to gitman-lock repo_path's own gitman.yml if it has any
    dep needing it -- either never locked at all, or locked but stale (drifted from
    what's installed). Both cases go through one combined prompt and one relock
    call, since gitman's lock() relocks the whole file in one shot regardless of
    which subset needed it. Declining never blocks the commit that follows. (A
    repo with nothing pending is still separately offered the top-level 'u' action
    in _review_repo whenever it owns a config file at all, unaffected by this.)
    """
    summary = _describe_pending_lock(configs, repo_path)
    if summary is None:
        return
    typer.echo(typer.style(
        f"This subproject's gitman.yml has dep(s) needing a lock: {summary}",
        fg="yellow"))
    if not typer.confirm(
        f"Run 'gitman lock' for {repo_path} before committing? This writes the "
        "currently-installed revisions into sources_locked.",
        default=False,
    ):
        return
    unlocked = _unlocked_deps(configs, repo_path)
    stale = _stale_locked_deps(configs, repo_path)
    _relock_config(repo, repo_path, unlocked + stale, configs)


def _review_repo(
    repo: Repo, repo_path: str, configs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Interact with the user about a single repo. Returns 'quit' to stop the whole review."""
    while True:
        typer.echo(typer.style(f"\n=== {repo_path} ===", bold=True, fg="cyan"))
        files = _changed_files(repo)
        if files:
            _print_file_list(files)
        else:
            typer.echo(typer.style("  Working tree clean.", fg="yellow"))

        stale_deps = _stale_locked_deps(configs, repo_path)
        if stale_deps:
            typer.echo(typer.style(
                f"⚠️  This subproject's gitman.yml lock is stale for {len(stale_deps)} "
                f"dep(s): {', '.join(stale_deps)}", fg="yellow"))

        has_config = _has_config(configs, repo_path)
        choices = ["c", "r", "d", "p"]
        if has_config:
            choices.append("u")
        choices += ["s", "q"]
        prompt_label = "commit / revert / diff / push" + (
            " / update-lock" if has_config else "") + " / skip / quit"
        default_choice = "u" if (stale_deps and not files) else ("p" if not files else "s")

        choice = typer.prompt(
            prompt_label,
            type=click.Choice(choices),
            default=default_choice,
            show_choices=True,
        )

        if choice == "q":
            return "quit"
        elif choice == "c":
            if not files:
                typer.echo("Nothing to commit.")
                continue
            if _commit_repo(repo, repo_path, files, configs):
                _push_repo(repo, repo_path)
            return None
        elif choice == "r":
            if not files:
                typer.echo("Nothing to revert.")
                continue
            if typer.confirm(
                f"Really revert {repo_path}? Changes will be stashed (recoverable) "
                "and the working tree reset to HEAD",
                default=False,
            ):
                _revert_repo(repo, repo_path)
            else:
                typer.echo("Cancelled.")
            return None
        elif choice == "d":
            if not files:
                typer.echo("No changed files to diff.")
                continue
            _show_diff_menu(repo, files)
            # loop back: re-show the (still up to date) file list and re-prompt
        elif choice == "p":
            _push_repo(repo, repo_path)
            if not files and not stale_deps:
                return None  # nothing else to decide for a clean, non-stale repo
            # working tree is still dirty, or there's a stale lock still pending:
            # loop back to decide its fate next
        elif choice == "u":
            _offer_manual_relock(repo, repo_path, configs)
            # loop back: gitman.yml is now a changed file in this same repo
        else:
            typer.echo(f"Skipped {repo_path}")
            return None


def _ordered_repo_paths(
    paths: Iterable[str], order: str, configs: Optional[Dict[str, Any]] = None
) -> List[str]:
    paths = set(paths)
    if order == "gitman" and configs:
        declared = [p for p in gitman_declared_order(configs) if p in paths and p != "."]
        remaining = sorted(p for p in paths if p not in declared and p != ".")
        ordered = declared + remaining
        if "." in paths:
            ordered.append(".")  # root last, same rationale as root-last
        return ordered

    ordered = sorted(paths)
    if order == "root-last" and "." in ordered:
        ordered.remove(".")
        ordered.append(".")
    return ordered


def run_review(
    root: Path,
    git_repos: Dict[str, Any],
    order: str = "root-last",
    configs: Optional[Dict[str, Any]] = None,
) -> None:
    """Shared review workflow, usable from `depman review` and `depman check -r`."""
    needs_review = {
        path: info
        for path, info in git_repos["repos"].items()
        if info["has_uncommitted"] or info["has_unpushed"] or _stale_locked_deps(configs, path)
    }
    if not needs_review:
        typer.echo(typer.style(
            "✅ No uncommitted changes, unpushed commits, or stale locks in any repo.",
            fg="green"))
        return

    typer.echo(typer.style(
        f"\nFound {len(needs_review)} repo(s) needing attention.",
        bold=True))

    for repo_path in _ordered_repo_paths(needs_review.keys(), order, configs):
        repo = Repo(root / repo_path)
        if _review_repo(repo, repo_path, configs) == "quit":
            typer.echo("Stopping review.")
            break

    typer.echo(typer.style("\nReview complete.", bold=True))


def review_cmd(
    ctx: typer.Context,
    use_cache: bool = typer.Option(
        False, "--use-cache", "-c",
        help="use the YAML snapshot from a previous 'check'/'list' scan instead of rescanning "
        "(faster, no network fetch; which repos are dirty may be stale if scanned a while ago).",
    ),
    order: str = typer.Option(
        "root-last", "--order", click_type=click.Choice(ORDER_CHOICES), help=ORDER_HELP,
    ),
    jobs: int = typer.Option(
        8, "--jobs", "-j",
        help="number of repos to fetch/scan concurrently during a live scan (default: 8).",
    ),
):
    """Semi-automatic review: commit or revert changes in each dirty repo, one at a time."""
    root: Path = ctx.obj["root"]
    # write_cache=False: a standalone `review` shouldn't leave new untracked cache
    # files in the very repo it's about to review.
    configs, git_repos = get_configs_and_repos(root, use_cache=use_cache, jobs=jobs, write_cache=False)
    run_review(root, git_repos, order=order, configs=configs)
