"""Tests for depman's git/config scanning and CLI commands."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from git import Repo

from depman.cli import cli
from depman.utils.configs import find_all_configs, find_all_git_repos


def _init_repo(path: Path) -> Repo:
    path.mkdir(parents=True, exist_ok=True)
    repo = Repo.init(path)
    (path / "README.md").write_text("hello\n")
    repo.index.add(["README.md"])
    repo.index.commit("initial commit")
    if repo.active_branch.name != "main":
        repo.git.branch("-m", "main")
    return repo


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A root git repo with one nested dep repo tracked by a root gitman.yml."""
    root = tmp_path / "project"
    dep = root / "deps" / "widget"

    _init_repo(root)
    _init_repo(dep)

    gitman_yml = {
        "location": "deps",
        "sources": [
            {"name": "widget", "repo": str(dep), "rev": "main"},
        ],
    }
    (root / "gitman.yml").write_text(yaml.safe_dump(gitman_yml))

    return root


def _dep_key() -> str:
    return str(Path("deps") / "widget")


def test_find_all_git_repos_finds_root_and_nested(project: Path):
    result = find_all_git_repos(project)
    assert "." in result["repos"]
    assert _dep_key() in result["repos"]


def test_find_all_configs_flattens_deps(project: Path):
    repos = find_all_git_repos(project)
    configs, repos = find_all_configs(project, repos)
    assert "." in configs["configs"]
    deps = configs["configs"]["."]["deps"]
    assert any(dep_info["name"] == "widget" for dep_info in deps.values())


def test_find_all_configs_handles_empty_gitman_yml(project: Path, capsys):
    """An empty gitman.yml (yaml.safe_load -> None) must not crash the scan."""
    nested = project / "nested"
    nested.mkdir()
    (nested / "gitman.yml").write_text("")

    repos = find_all_git_repos(project)
    configs, repos = find_all_configs(project, repos)

    assert "nested" in configs["configs"]
    assert configs["configs"]["nested"]["deps"] == {}
    assert "Error loading" not in capsys.readouterr().out


def test_find_all_configs_defaults_location_when_missing(project: Path):
    """A gitman.yml without a 'location' key should fall back to gitman's own default."""
    nested = project / "nested2"
    nested.mkdir()
    (nested / "gitman.yml").write_text(yaml.safe_dump({"sources": []}))

    repos = find_all_git_repos(project)
    configs, repos = find_all_configs(project, repos)

    assert configs["configs"]["nested2"]["location"] == "gitman_sources"


def test_cli_list_runs_against_scanned_project(project: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "list"])
    assert result.exit_code == 0, result.output
    assert "Repos Summary" in result.output


def test_cli_check_runs_against_scanned_project(project: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "check"])
    assert result.exit_code == 0, result.output
    assert "Git Repos Status" in result.output


def test_cli_check_json_prints_clean_json_on_stdout(project: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "check", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)
    assert "." in payload["repos"]["repos"]
    assert "." in payload["configs"]["configs"]
    # diagnostic/progress noise must not leak onto stdout
    assert "find_all_git_repos" not in result.stdout
    assert "Dumped" not in result.stdout
    # ...but it's still there, just on stderr
    assert "find_all_git_repos" in result.stderr


def test_cli_list_json_prints_clean_json_on_stdout(project: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "list", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)
    assert "." in payload["repos"]["repos"]
    assert "." in payload["configs"]["configs"]


def test_review_reports_clean_when_no_changes(project: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"])
    assert result.exit_code == 0, result.output
    assert "No uncommitted changes" in result.output


def test_review_commit_commits_dirty_repo(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    # c (commit) -> confirm file list (blank = yes) -> commit message (blank = default)
    # (project fixture has no 'origin' remote, so the post-commit push attempt
    # just prints an error and consumes no further input)
    result = runner.invoke(cli, ["--root", str(project), "review"], input="c\n\n\n")
    assert result.exit_code == 0, result.output
    assert "About to commit:" in result.output
    assert "Committed ." in result.output
    assert not Repo(project).is_dirty(untracked_files=True)


def test_review_commit_shows_new_files_and_can_be_cancelled(project: Path):
    (project / "README.md").write_text("changed\n")
    (project / "brand_new.txt").write_text("new stuff\n")
    runner = CliRunner()
    # c (commit) -> decline the confirm
    result = runner.invoke(cli, ["--root", str(project), "review"], input="c\nn\n")
    assert result.exit_code == 0, result.output
    assert "new/untracked file" in result.output
    assert "brand_new.txt" in result.output
    assert "Cancelled" in result.output
    assert Repo(project).is_dirty(untracked_files=True)


def test_review_revert_stashes_dirty_repo(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"], input="r\ny\n")
    assert result.exit_code == 0, result.output
    assert "Reverted ." in result.output
    repo = Repo(project)
    # The tracked change is reverted; a nested repo dir ("deps/") is left
    # untouched by `git stash -u` (git never sweeps up embedded .git dirs),
    # so check the same has_uncommitted definition depman itself uses.
    assert not repo.is_dirty()
    assert "depman-revert-safety-" in repo.git.stash("list")


def test_review_revert_cancelled_keeps_changes(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"], input="r\nn\n")
    assert result.exit_code == 0, result.output
    assert "Cancelled" in result.output
    assert Repo(project).is_dirty(untracked_files=True)


def test_review_skip_leaves_repo_dirty(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"], input="s\n")
    assert result.exit_code == 0, result.output
    assert Repo(project).is_dirty(untracked_files=True)


def test_review_shows_file_list_not_full_diff_by_default(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"], input="s\n")
    assert result.exit_code == 0, result.output
    assert "Changed files:" in result.output
    assert "README.md" in result.output
    assert "diff --git" not in result.output


def test_review_diff_then_commit(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    # d (diff) -> blank (all files) -> c (commit) -> confirm blank -> message blank
    # (project fixture has no 'origin' remote, so the post-commit push attempt
    # just prints an error and consumes no further input)
    result = runner.invoke(cli, ["--root", str(project), "review"], input="d\n\nc\n\n\n")
    assert result.exit_code == 0, result.output
    assert "diff --git" in result.output
    assert "Committed ." in result.output


def test_review_diff_select_file_by_number(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"], input="d\n0\ns\n")
    assert result.exit_code == 0, result.output
    assert "diff --git" in result.output
    assert "Skipped ." in result.output


def test_review_use_cache_skips_rescan(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    # populate the scan cache first, the way `depman check`/`list` normally would
    pre = runner.invoke(cli, ["--root", str(project), "check"])
    assert pre.exit_code == 0, pre.output
    assert (project / ".cache_git_repos.yaml").exists()

    result = runner.invoke(cli, ["--root", str(project), "review", "-c"], input="c\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Loaded cached git status" in result.output
    assert "Committed ." in result.output
    assert not Repo(project).is_dirty()


def test_review_push_without_remote_reports_error(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"], input="p\ns\n")
    assert result.exit_code == 0, result.output
    assert "no 'origin' remote" in result.output
    assert "Skipped ." in result.output


def test_review_push_pushes_existing_unpushed_commit(tmp_path: Path):
    root = tmp_path / "proj"
    bare = tmp_path / "origin.git"
    Repo.init(bare, bare=True)
    repo = _init_repo(root)
    repo.create_remote("origin", str(bare))
    repo.git.push("origin", "main")
    # a local commit the remote doesn't have yet
    (root / "README.md").write_text("second commit\n")
    repo.git.add(A=True)
    repo.index.commit("second commit")
    # plus an uncommitted change so the repo shows up in review at all
    (root / "README.md").write_text("second commit\nworking change\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "review"], input="p\ny\ns\n")
    assert result.exit_code == 0, result.output
    assert "Commits to push (1):" in result.output
    assert "second commit" in result.output
    assert "Pushed ." in result.output
    assert Repo(bare).commit("main").message.strip() == "second commit"


def test_review_commit_then_push_now(tmp_path: Path):
    root = tmp_path / "proj"
    bare = tmp_path / "origin.git"
    Repo.init(bare, bare=True)
    repo = _init_repo(root)
    repo.create_remote("origin", str(bare))
    repo.git.push("origin", "main")
    (root / "README.md").write_text("changed\n")

    runner = CliRunner()
    # c -> confirm blank(yes) -> message blank -> push confirm "y"
    result = runner.invoke(cli, ["--root", str(root), "review"], input="c\n\n\ny\n")
    assert result.exit_code == 0, result.output
    assert "Committed ." in result.output
    assert "Commits to push (1):" in result.output
    assert "Pushed ." in result.output
    assert Repo(bare).commit("main").message.strip() == "wip: ."


def test_review_includes_clean_repo_with_only_unpushed_commits(tmp_path: Path):
    """A repo with a clean working tree but unpushed commits must still show up in review."""
    root = tmp_path / "proj"
    bare = tmp_path / "origin.git"
    Repo.init(bare, bare=True)
    repo = _init_repo(root)
    repo.create_remote("origin", str(bare))
    repo.git.push("origin", "main")
    (root / "README.md").write_text("second commit\n")
    repo.git.add(A=True)
    repo.index.commit("second commit")
    assert not repo.is_dirty(untracked_files=True)  # working tree is clean

    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "review"], input="\ny\n")
    assert result.exit_code == 0, result.output
    assert "Working tree clean" in result.output
    # default choice for a clean-but-unpushed repo is 'p', so blank input pushes it
    assert "Commits to push (1):" in result.output
    assert "Pushed ." in result.output
    assert Repo(bare).commit("main").message.strip() == "second commit"


def test_review_clean_repo_commit_and_revert_are_no_ops(tmp_path: Path):
    root = tmp_path / "proj"
    bare = tmp_path / "origin.git"
    Repo.init(bare, bare=True)
    repo = _init_repo(root)
    repo.create_remote("origin", str(bare))
    repo.git.push("origin", "main")
    (root / "README.md").write_text("second commit\n")
    repo.git.add(A=True)
    repo.index.commit("second commit")

    runner = CliRunner()
    # c -> "Nothing to commit" -> loop back -> r -> "Nothing to revert" -> loop back -> s (skip)
    result = runner.invoke(cli, ["--root", str(root), "review"], input="c\nr\ns\n")
    assert result.exit_code == 0, result.output
    assert "Nothing to commit." in result.output
    assert "Nothing to revert." in result.output
    assert "Skipped ." in result.output
    # commit/revert were no-ops and we never pushed, so the bare remote
    # still only has the initial commit
    assert [c.message.strip() for c in Repo(bare).iter_commits("main")] == ["initial commit"]


def test_review_order_root_last_processes_dep_before_root(project: Path):
    (project / "README.md").write_text("root changed\n")
    (project / "deps" / "widget" / "README.md").write_text("widget changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "review"], input="s\ns\n")
    assert result.exit_code == 0, result.output
    dep_key = _dep_key()
    assert result.output.index(f"=== {dep_key} ===") < result.output.index("=== . ===")


def test_review_order_root_first_processes_root_before_dep(project: Path):
    (project / "README.md").write_text("root changed\n")
    (project / "deps" / "widget" / "README.md").write_text("widget changed\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(project), "review", "--order", "root-first"], input="s\ns\n"
    )
    assert result.exit_code == 0, result.output
    dep_key = _dep_key()
    assert result.output.index("=== . ===") < result.output.index(f"=== {dep_key} ===")


def test_check_review_flag_shows_table_then_enters_review(project: Path):
    (project / "README.md").write_text("changed\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(project), "check", "-r"], input="s\n")
    assert result.exit_code == 0, result.output
    assert "Git Repos Status" in result.output
    assert "Changed files:" in result.output
