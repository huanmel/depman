"""
Config backend interface: a pluggable source of declared/locked dependency
revisions. Only `gitman.yml` is implemented today; git submodules could
implement this same interface later (GitPython's Submodule API already maps
onto the same shape -- .hexsha is the recorded/locked SHA, .module().head is
the actually-installed one).

Everything downstream of `find_all_configs` (repo scanning, cross-referencing,
ordering, display, the `review` workflow, `hooks`) only depends on every
config entry having the same `deps[dep_path] = {name, rev, rev_locked,
rev_installed, ...}` shape -- it never needs to know which backend produced
an entry.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class ConfigBackend(ABC):
    name: str

    @abstractmethod
    def discover(self, root: Path) -> List[Path]:
        """Find every config location for this backend under root."""

    @abstractmethod
    def parse_one(self, config_path: Path, root: Path, repos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse one discovered config into the standard per-config entry shape:
        {'project_root': str, 'location': str, 'rev_installed': str or None,
         'deps': {dep_path: {'name', 'repo', 'rev', 'rev_locked', 'rev_installed', 'path'}}}
        `repos` is the already-scanned {'repos': {...}} map (path -> git status
        entry); may be mutated to cross-reference a dep's `used_in_configs`.
        """

    @abstractmethod
    def relock(self, config_dir: Path, dep_names: List[str]) -> None:
        """Sync the lock/pointer for dep_names (or all, if empty) to whatever
        revisions are currently installed in config_dir."""
