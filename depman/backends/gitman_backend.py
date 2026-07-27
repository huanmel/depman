"""
Gitman backend: parses gitman.yml (declared `sources` + locked `sources_locked`)
and relocks via gitman's own `lock()` API. This is the exact logic that used
to live inline in configs.py's find_all_configs / review.py's relock action,
moved here unchanged behind the ConfigBackend interface.
"""
from pathlib import Path
from typing import Any, Dict, List

import yaml
from gitman import lock as gitman_lock

from depman import CONFIG_NAME
from depman.backends.base import ConfigBackend


class GitmanBackend(ConfigBackend):
    name = "gitman"

    def discover(self, root: Path) -> List[Path]:
        return list(root.glob(f"**/{CONFIG_NAME}"))

    def parse_one(self, config_path: Path, root: Path, repos: Dict[str, Any]) -> Dict[str, Any]:
        project_root = config_path.parent
        with open(config_path) as f:
            content = yaml.safe_load(f)
        if not isinstance(content, dict):
            # empty file (safe_load -> None) or malformed content (not a mapping)
            content = {}

        location = content.get("location", "gitman_sources")  # gitman's own default
        deps = {}
        deps_locked = {}

        for req in content.get("sources_locked", []):
            name = req.get("name", Path(req["repo"]).name)
            dep_path = project_root / location / name
            deps_locked[str(dep_path.relative_to(root))] = {
                "name": name,
                "repo": req["repo"],
                "rev": req["rev"]
            }
        for req in content.get("sources", []):
            name = req.get("name", Path(req["repo"]).name)
            dep_path = project_root / location / name
            dep_path_rel = str(dep_path.relative_to(root))
            rev_installed = repos.get(dep_path_rel, {}).get("rev") if dep_path_rel in repos else None
            rev_locked = deps_locked[dep_path_rel]["rev"] if dep_path_rel in deps_locked else None
            rev = req["rev"]
            if rev_installed:
                proj_root_relative = str(project_root.relative_to(root))
                repos[dep_path_rel]["used_in_configs"].update(
                    {proj_root_relative: rev_locked if rev_locked else rev_installed})

            deps[dep_path_rel] = {
                "name": name,
                "repo": req["repo"],
                "rev": rev,
                "rev_locked": rev_locked,
                "rev_installed": rev_installed,
                "path": dep_path_rel
            }
        project_root_short = str(project_root.relative_to(root))
        rev_installed = repos.get(project_root_short, {}).get("rev")

        return {
            "project_root": project_root_short,
            "config_file": str(config_path.name),
            "location": location,
            "rev_installed": rev_installed,
            "deps": deps,
        }

    def relock(self, config_dir: Path, dep_names: List[str]) -> None:
        gitman_lock(root=config_dir)
