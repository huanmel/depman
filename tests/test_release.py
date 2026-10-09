"""Tests for `depman release`."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from git import Repo

from depman.cli import cli
from depman.commands.release import parse_version, resolve_version


def test_resolve_version_bumps():
    assert resolve_version("fix", (0, 1, 2)) == (0, 1, 3)
    assert resolve_version("patch", (0, 1, 2)) == (0, 1, 3)
    assert resolve_version("minor", (0, 1, 2)) == (0, 2, 0)
    assert resolve_version("major", (0, 1, 2)) == (1, 0, 0)


def test_resolve_version_without_previous_starts_at_zero():
    assert resolve_version("fix", None) == (0, 0, 1)
    assert resolve_version("minor", None) == (0, 1, 0)
    assert resolve_version("major", None) == (1, 0, 0)


def test_resolve_version_explicit_and_invalid():
    assert resolve_version("v2.0.5", (0, 1, 2)) == (2, 0, 5)
    assert resolve_version("2.0.5", None) == (2, 0, 5)
    with pytest.raises(ValueError):
        resolve_version("huge", None)
    assert parse_version("v1.2") is None


@pytest.fixture
def released_project(tmp_path: Path):
    """A repo on 'main' with a bare 'origin' and one existing release tag v0.1.2."""
    bare = tmp_path / "origin.git"
    Repo.init(bare, bare=True)
    root = tmp_path / "project"
    root.mkdir()
    repo = Repo.init(root)
    (root / "VERSION").write_text("0.1.2\n")
    repo.git.add(A=True)
    repo.git.commit("-m", "initial")
    repo.git.branch("-m", "main")
    repo.create_remote("origin", str(bare))
    repo.git.tag("-a", "v0.1.2", "-m", "Release v0.1.2")
    repo.git.push("origin", "main", "--follow-tags")
    return root, Repo(bare)


def _release(root: Path, args, answers: str):
    return CliRunner().invoke(cli, ["--root", str(root), "release", *args], input=answers)


def test_release_bump_end_to_end(released_project):
    root, bare = released_project
    script = "python -c \"open('VERSION','w').write('{version}')\""
    # proceed, commit message (default), commit all, push
    result = _release(root, ["fix", "--version-script", script], "y\n\ny\ny\n")
    assert result.exit_code == 0, result.output
    assert "v0.1.3" in result.output
    assert (root / "VERSION").read_text() == "0.1.3"
    repo = Repo(root)
    assert repo.head.commit.message.strip() == "Release v0.1.3"
    assert repo.tags["v0.1.3"].commit == repo.head.commit
    assert bare.tags["v0.1.3"].commit.hexsha == repo.head.commit.hexsha
    assert bare.refs["main"].commit.hexsha == repo.head.commit.hexsha


def test_release_dry_run_changes_nothing(released_project):
    root, bare = released_project
    result = _release(root, ["minor", "--dry-run"], "")
    assert result.exit_code == 0
    assert "v0.2.0" in result.output
    assert [t.name for t in Repo(root).tags] == ["v0.1.2"]


def test_release_declining_confirmation_aborts(released_project):
    root, _ = released_project
    result = _release(root, ["major", "--version-script", "echo nope > touched.txt"], "n\n")
    assert result.exit_code == 1
    assert not (root / "touched.txt").exists()


def test_release_existing_tag_requires_force(released_project):
    root, _ = released_project
    result = _release(root, ["v0.1.2"], "")
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_release_force_tag_moves_tag_locally_and_remotely(released_project):
    root, bare = released_project
    (root / "fix.txt").write_text("late fix\n")
    result = _release(root, ["v0.1.2", "--force-tag"], "y\n\ny\ny\n")
    assert result.exit_code == 0, result.output
    repo = Repo(root)
    assert repo.tags["v0.1.2"].commit == repo.head.commit
    assert bare.tags["v0.1.2"].commit.hexsha == repo.head.commit.hexsha


def test_release_failing_script_aborts_before_commit(released_project):
    root, _ = released_project
    head = Repo(root).head.commit.hexsha
    result = _release(root, ["fix", "--version-script", "exit 3"], "y\n")
    assert result.exit_code == 1
    assert Repo(root).head.commit.hexsha == head


def test_release_uses_config_file_scripts(released_project):
    root, _ = released_project
    (root / ".depman.yml").write_text(
        "release:\n  version_scripts:\n    - \"python -c \\\"open('VERSION','w').write('{tag}')\\\"\"\n")
    result = _release(root, ["minor"], "y\n\ny\ny\n")
    assert result.exit_code == 0, result.output
    assert (root / "VERSION").read_text() == "v0.2.0"
