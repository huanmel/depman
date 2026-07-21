"""Tests for depman's git/config scanning and CLI commands."""

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
