"""
Update checker using Gitman API + GitPython.
"""

import click
from pathlib import Path
from typing import List, Dict, Any, Optional
from git import Repo, GitCommandError
from gitman.models import Config
import yaml  # For manual YAML loads
import time
from functools import wraps
import sys
from datetime import datetime

CONFIG_NAME = 'gitman.yml'  # Global for config filename
CACHE_GIT_REPOS = ".cache_git_repos.yaml"
CACHE_CONFIGS = ".cache_configs.yaml"

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function: {func.__name__}{args} {kwargs} \nTimeit: {total_time:.4f} sec')
        return result
    return timeit_wrapper

def find_nested_configs(base_path: Path, depth: int = float("inf")) -> List[Path]:
    """Find nested .gitman.yml files (simple walker; max depth to avoid cycles)."""
    configs = []
    for config_path in base_path.glob(f"**/{CONFIG_NAME}"):
        rel_depth = len(config_path.relative_to(base_path).parts) - 1
        if depth > 0 and rel_depth >= depth:
            continue
        configs.append(config_path)
    return configs

@timeit
def find_all_configs(root: Path) -> Dict[str, Any]:
    """
    Load all .gitman.yml configs under root, flatten deps into structure.
    Returns: {'configs': {config_path: str: {'project_root': Path, 'location': str, 'deps': List[Dict{'name': str, 'repo': str, 'rev': str, 'path': Path}]}}}
    """
    configs = {}
    for config_path in root.glob(f"**/{CONFIG_NAME}"):
        project_root = config_path.parent
        try:
            with open(config_path) as f:
                content = yaml.safe_load(f)
            
            location = content.get("location")
            deps = []
            # Flatten requirements
            for req in content.get("sources", []):
                name = req.get("name", Path(req["repo"]).name)
                dep_path = project_root / location / name
                
                deps.append({
                    "name": name,
                    "repo": req["repo"],
                    "rev": req["rev"],
                    "path": str(dep_path.relative_to(root))
                })
            # # Flatten groups
            # for group_name, group_reqs in content.get("groups", {}).items():
            #     for req in group_reqs:
            #         name = req.get("name", Path(req["repo"]).name)
            #         dep_path = project_root / location / name
            #         deps.append({
            #             "name": name,
            #             "repo": req["repo"],
            #             "rev": req["rev"],
            #             "path": dep_path
            #         })
            project_root_short = project_root.relative_to(root)
            configs[str(project_root_short)] = {
                "project_root": str(project_root_short),
                "config_file": str(config_path.name),
                "location": location,
                "deps": deps
            }
        except Exception as e:
            click.echo(click.style(f"Error loading {config_path}: {e}", fg="red"))
    
    res = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configs": configs
    }
    return res


@timeit
def find_all_git_repos(root: Path) -> Dict[str, Any]:
    """
    Find all Git repos under root, fetch upstream, and detect updates/uncommitted/unpushed.
    Returns: {
        'repos': {
            repo_path: str (relative): {
                'project_root': Path,
                'revision': str (full HEAD SHA),
                'short_revision': str (7 chars),
                'remote_url': str or None,
                'tags': List[str],
                'current_tag': str or None,
                'current_branch': str,
                'has_update': bool (on current branch post-fetch),
                'update_details': Dict{'branch': str, 'latest_hash': str, 'datetime': str, 'message': str} or None,
                'has_update_main': bool (if current != 'main'),
                'update_details_main': Dict[...] or None,
                'has_uncommitted': bool,
                'uncommitted_files': List[str] (relative paths if dirty),
                'has_unpushed': bool (commits ahead of origin),
                'unpushed_count': int
            }
        }
    }
    """
    
    repos = {}
    for repo_path in root.rglob(".git"):
        if repo_path.is_dir():
            git_root = repo_path.parent
            try:
                repo = Repo(git_root)
                revision = repo.head.commit.hexsha
                short_revision = revision[:7]
                remote_url = repo.remotes.origin.url if hasattr(repo.remotes, 'origin') else None
                tags = [t.name for t in repo.tags]
                current_tag = (
                    repo.head.ref.name.replace("refs/tags/", "")
                    if repo.head.ref and repo.head.ref.name.startswith("refs/tags/")
                    else None
                )
                current_branch = repo.active_branch.name if repo.head.is_detached else repo.head.ref.name.split('/')[-1]

                # Fetch upstream (origin)
                if hasattr(repo.remotes, 'origin'):
                    repo.remotes.origin.fetch()
                else:
                    current_branch = None  # No updates/unpushed possible

                # Check updates on current branch
                has_update = False
                update_details = None
                if current_branch:
                    try:
                        origin_ref = repo.refs[f"origin/{current_branch}"]
                        latest_hash = origin_ref.commit.hexsha
                        if revision != latest_hash:
                            has_update = True
                            commit = origin_ref.commit
                            update_details = {
                                "branch": current_branch,
                                "latest_hash": latest_hash,
                                "datetime": commit.authored_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                "message": commit.message.split('\n')[0]  # First line
                            }
                    except IndexError:
                        pass  # No origin/<branch> ref

                # Additional check for main if current != 'main'
                has_update_main = False
                update_details_main = None
                if current_branch and current_branch != "main":
                    for main_branch in ["main", "master"]:  # Fallback
                        try:
                            main_ref = repo.refs[main_branch]
                            origin_main_ref = repo.refs[f"origin/{main_branch}"]
                            if main_ref.commit.hexsha != origin_main_ref.commit.hexsha:
                                has_update_main = True
                                commit = origin_main_ref.commit
                                update_details_main = {
                                    "branch": main_branch,
                                    "latest_hash": origin_main_ref.commit.hexsha,
                                    "datetime": commit.authored_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                    "message": commit.message.split('\n')[0]
                                }
                                break
                        except (IndexError, AttributeError):
                            continue

                # New: Uncommitted changes
                has_uncommitted = repo.is_dirty()
                uncommitted_files = []
                if has_uncommitted:
                    # Untracked
                    uncommitted_files.extend(repo.untracked_files)
                    # Staged/unstaged diffs
                    uncommitted_files.extend([diff.a_path for diff in repo.index.diff(None)])
                    uncommitted_files.extend([diff.a_path for diff in repo.index.diff("HEAD")])

                # New: Unpushed commits (local ahead)
                has_unpushed = False
                unpushed_count = 0
                if current_branch and hasattr(repo.remotes, 'origin'):
                    try:
                        unpushed_commits = list(repo.iter_commits(f"origin/{current_branch}..HEAD"))
                        unpushed_count = len(unpushed_commits)
                        has_unpushed = unpushed_count > 0
                    except IndexError:
                        pass  # No origin ref

                project_root_short = git_root.relative_to(root)
                repos[str(project_root_short)] = {
                    "name": str(project_root_short.name) if project_root_short.name else ".",
                    "project_root": str(project_root_short),
                    "revision": revision,
                    "short_revision": short_revision,
                    "remote_url": remote_url,
                    "tags": tags,
                    "current_tag": current_tag,
                    "current_branch": current_branch,
                    "has_update": has_update,
                    "update_details": update_details,
                    "has_update_main": has_update_main,
                    "update_details_main": update_details_main,
                    "has_uncommitted": has_uncommitted,
                    "uncommitted_files": uncommitted_files,
                    "has_unpushed": has_unpushed,
                    "unpushed_count": unpushed_count
                }
            except Exception as e:
                click.echo(click.style(f"  {repo_path}: Error ({e})", fg="red"))
    res = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repos": repos
    }
    return res


def is_git_repo(path: Path) -> bool:
    """Check if path is a Git repo."""
    try:
        Repo(str(path), search_parent_directories=False)
        return True
    except:
        return False


def print_project_tree(config: Config, root_path: Path, prefix: str = ""):
    """Print ASCII tree of project deps (requirements + groups)."""
    location = config.content.get("location", "requirements")
    click.echo(f"{prefix}├── Location: {location}")
    
    def _print_node(items: List[Dict], node_prefix: str = "│   ", is_last: bool = True):
        for i, req in enumerate(items):
            is_last_item = i == len(items) - 1
            conn = "└── " if is_last_item else "├── "
            click.echo(f"{node_prefix}{conn}{req.get('name', Path(req['repo']).name)} ({req['rev']})")
            if is_last_item:
                node_prefix = "    "
    
    # Requirements
    if config.requirements:
        click.echo(f"{prefix}├── Requirements:")
        _print_node(config.requirements, prefix="│   ", is_last=False)
    
    # Groups
    for group_name, group_reqs in config.groups.items():
        is_last_group = group_name == list(config.groups.keys())[-1]
        conn = "└── " if is_last_group else "├── "
        click.echo(f"{prefix}{conn}Group: {group_name}")
        _print_node(group_reqs, prefix=prefix + ("    " if is_last_group else "│   "))


def get_all_requirements(config: Config, recursive: bool = False) -> List[Dict[str, Any]]:
    """Flatten requirements + groups (no public get_dependencies, so manual)."""
    reqs = config.requirements[:]
    for group_name, group_reqs in config.groups.items():
        reqs.extend(group_reqs)
    if recursive:
        location = Path(config.directory or ".") / config.content.get("location", "requirements")
        for nested_config_path in find_nested_configs(location):
            nested_config = Config.load(nested_config_path)
            reqs.extend(get_all_requirements(nested_config, recursive=False))  # Avoid infinite recursion
    return reqs


def scan_gitman_projects(root: Path, scan_depth: int = float("inf")) -> List[tuple[Path, Config]]:
    """Scan for Gitman projects (now uses load_all_configs for loading)."""
    projects = []
    loaded = find_all_configs(root)
    for config_path_str, data in loaded["configs"].items():
        config_path = Path(config_path_str)
        project_root = data["project_root"]
        if is_git_repo(project_root):
            try:
                config = Config.load(config_path)
                projects.append((project_root, config))
            except Exception as e:
                click.echo(click.style(f"Error loading {config_path}: {e}", fg="red"))
    return projects


@click.command()
# @click.option("--recursive", is_flag=True, help="Flatten deps recursively for updates (legacy).")
# @click.option("--scan-depth", type=int, default=None, help="Max recursion depth for scanning (default: unlimited).")
@click.option("--use-cache",  is_flag=True, default=False, help="use cached YAML files instead of live scan (path to root).")
@click.pass_context
def check_updates_cmd(ctx: click.Context, use_cache: bool): #, recursive: bool, scan_depth: Optional[int]
    """Check for upstream updates in Gitman dependencies (now scans all projects)."""
    root = ctx.obj["root"]
    # if scan_depth is None:
    #     scan_depth = float("inf")

    if use_cache:
        loaded_configs = yaml.safe_load(open(root/ CACHE_CONFIGS))
        print(f"✅ Loaded cached configs from {str(root/ CACHE_CONFIGS)}") 
        git_repos = yaml.safe_load(open( root / CACHE_GIT_REPOS))
        print(f"✅ Loaded cached git status from {str(root/ CACHE_GIT_REPOS)}") 
        
    else:
        loaded_configs = find_all_configs(root)
        cache_conf_file=root / CACHE_CONFIGS
        with open(cache_conf_file, "w") as f:
            yaml.safe_dump(loaded_configs, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Dumped loaded_configs to {cache_conf_file}")
    
    
    
        git_repos = find_all_git_repos(root)
        num_repos = len(git_repos["repos"])
        print(f"✅ find_all_git_repos: Found {num_repos} repos")
        cache_repos_file=root / CACHE_GIT_REPOS
        with open(cache_repos_file, "w") as f:
            yaml.safe_dump(git_repos, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Dumped git_repos to {cache_repos_file}")
    
    
    for key, val in git_repos["repos"].items():
        if val["has_uncommitted"]:
            click.echo(click.style(f"⚠️  Repo {key} has uncommitted changes: {val['uncommitted_files']}", fg="yellow"))
        if val["has_unpushed"]:
            click.echo(click.style(f"⚠️  Repo {key} has {val['unpushed_count']} unpushed commits", fg="yellow"))
        if val["has_update"]:
            details = val["update_details"]
            click.echo(click.style(f"⚠️  Repo {key} has updates on branch {details['branch']}: {details['latest_hash'][:7]} - {details['message']}", fg="yellow"))
        if val["has_update_main"]:
            details = val["update_details_main"]
            click.echo(click.style(f"⚠️  Repo {key} has updates on main branch {details['branch']}: {details['latest_hash'][:7]} - {details['message']}", fg="yellow"))

    return # Early return to avoid full project scan for now
    for project_root, config in projects:
        click.echo(click.style(f"\n=== Project: {project_root} ===", bold=True))
        print_project_tree(config, project_root)

        location = project_root / config.content.get("location", "requirements")
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
                        status = click.style(f"Update avail. ({short_current} → {short_latest})", fg="red")
                    else:
                        status = click.style(f"Mismatch ({short_latest})", fg="yellow")

                table.add_row([name, status, repo_url])

            except GitCommandError as e:
                status = click.style(f"Git error ({e})", fg="red")
                table.add_row([name, status, repo_url])
            except KeyError as e:
                status = click.style(f"Missing key ({e})", fg="yellow")
                table.add_row([name, status, repo_url])

        click.echo(table)
        if has_updates:
            click.echo(click.style("⚠️  Updates available—run 'depman gm update' to apply.", fg="yellow"))


if __name__ == "__main__":
    """Quick debug tests—run: python -m depman.checker [--test-path /path/to/project]"""
    import tempfile
    import argparse  # Simple arg parse for path (lightweight)
    
    parser = argparse.ArgumentParser(description="Debug Checker Functions")
    parser.add_argument("--test-path", type=str, default=None, help="Path to real Gitman project for full test (overrides mocks)")
    args = parser.parse_args()
    
    print("=== Debug Tests for Checker Functions ===")
    if args.test_path:
        # Full real-project test
        root = Path(args.test_path).resolve()
        if not root.exists():
            raise ValueError(f"Test path {root} does not exist.")
        
        print(f"🔍 Testing with real project: {root}")
        loaded_configs = find_all_configs(root)
        
        # New: Test load_all_configs
        
        cache_conf_file=root / CACHE_CONFIGS
        with open(cache_conf_file, "w") as f:
            yaml.safe_dump(loaded_configs, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Dumped loaded_configs to {cache_conf_file}")

        
        assert "configs" in loaded_configs, "No 'configs' key"
        num_configs = len(loaded_configs["configs"])
        assert num_configs > 0, f"No configs found in {root}"
        print(f"✅ load_all_configs: Loaded {num_configs} configs")
        print("Sample config structure:", list(loaded_configs["configs"].keys())[:1])  # First key
        if num_configs > 0:
            sample_deps = loaded_configs["configs"]['.']["deps"]
            print(f"  Sample deps: {len(sample_deps)} (e.g., {sample_deps[0] if sample_deps else 'none'})")
        
        # New: Test find_all_git_repos
        git_repos = find_all_git_repos(root)
        assert "repos" in git_repos, "No 'repos' key"
        num_repos = len(git_repos["repos"])
        print(f"✅ find_all_git_repos: Found {num_repos} repos")
        if num_repos > 0:
            sample_repo = list(git_repos["repos"].values())[0]
            print(f"  Sample repo info: rev={sample_repo['revision'][:8]}, remote={sample_repo['remote_url']}, tags={len(sample_repo['tags'])}")
        
        cache_repos_file=root / CACHE_GIT_REPOS
        with open(cache_repos_file, "w") as f:
            yaml.safe_dump(git_repos, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Dumped git_repos to {cache_repos_file}")
        sys.exit(0)
        
        # # Existing tests (scan, tree, etc.)
        # configs = find_nested_configs(test_root, depth=float("inf"))
        # assert len(configs) > 0, f"No .gitman.yml found in {test_root}"
        # print(f"✅ find_nested_configs: Found {len(configs)} files")
        
        # projects = scan_gitman_projects(test_root)
        # assert len(projects) > 0, f"No Gitman projects in {test_root}"
        # print(f"✅ scan_gitman_projects: Found {len(projects)} projects")
        
        # project_root, config = projects[0]
        # print(f"\n--- Real Tree Output for {project_root} ---")
        # print_project_tree(config, project_root)
        # all_reqs = get_all_requirements(config)
        # assert len(all_reqs) > 0, "No requirements in project"
        # print(f"✅ print_project_tree & get_all_requirements: Tree printed, {len(all_reqs)} flattened reqs")
        
        # # End-to-end update check (limited)
        # location = project_root / config.content.get("location", "requirements")
        # has_updates = False
        # for req in all_reqs[:3]:
        #     name = req.get("name", Path(req["repo"]).name)
        #     repo_url = req["repo"]
        #     target_rev = req["rev"]
        #     dep_path = location / name
        #     if not dep_path.exists():
        #         print(f"  {name}: Not installed (skipping)")
        #         continue
        #     try:
        #         repo = Repo(dep_path)
        #         repo.remotes.origin.fetch()
        #         current_sha = repo.head.commit.hexsha
        #         try:
        #             origin_ref = repo.refs[f"origin/{target_rev}"]
        #             latest_sha = origin_ref.commit.hexsha
        #         except IndexError:
        #             latest_sha = repo.rev_parse(target_rev).hexsha
        #         if current_sha != latest_sha:
        #             has_updates = True
        #             print(f"  {name}: Update detected! ({current_sha[:8]} → {latest_sha[:8]})")
        #         else:
        #             print(f"  {name}: Up to date ({current_sha[:8]})")
        #     except Exception as e:
        #         print(f"  {name}: Error ({e})")
        # print(f"✅ End-to-end update check: Ran on {min(3, len(all_reqs))} deps, updates: {has_updates}")
        
        print("\n=== Real Project Debug Tests Passed! ===")
    else:
        # Fallback mock tests (updated to use new functions)
        print("🧪 Running mock tests (use --test-path for real project)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create nested .gitman.yml
            (tmp_path / "proj1" / CONFIG_NAME).write_text(yaml.dump({
                "requirements": [{"name": "dep1", "repo": "https://ex/dep1", "rev": "main"}]
            }))
            (tmp_path / "proj1" / "sub" / CONFIG_NAME).write_text(yaml.dump({
                "groups": {"g1": [{"name": "dep2", "repo": "https://ex/dep2", "rev": "v1"}]}
            }))
            loaded = find_all_configs(tmp_path)
            assert len(loaded["configs"]) == 2
            print(f"✅ load_all_configs (mock): {len(loaded['configs'])} configs")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            proj1 = tmp_path / "proj1"
            proj1.mkdir()
            (proj1 / ".git").mkdir()  # Simple mock; full Repo needs init/commit in real
            assert is_git_repo(proj1) is False  # Empty .git not valid
            print("✅ is_git_repo: Handled invalid (for mock)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / CONFIG_NAME
            config_path.write_text(yaml.dump({
                "requirements": [{"name": "req1", "repo": "https://ex/req1", "rev": "main"}],
                "groups": {"testgroup": [{"name": "gdep1", "repo": "https://ex/gdep1", "rev": "develop"}]}
            }))
            (tmp_path / ".git").mkdir()
            config = Config.load(config_path)
            print("\n--- Sample Tree Output ---")
            print_project_tree(config, tmp_path)
            all_reqs = get_all_requirements(config)
            assert len(all_reqs) == 2
            print(f"✅ print_project_tree & get_all_requirements (mock): {len(all_reqs)} reqs")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            proj_dir = tmp_path / "testproj"
            proj_dir.mkdir()
            (proj_dir / ".git").mkdir()
            (proj_dir / CONFIG_NAME).write_text(yaml.dump({
                "requirements": [{"name": "testdep", "repo": "https://ex/test", "rev": "main"}]
            }))
            projects = scan_gitman_projects(tmp_path)
            assert len(projects) == 0  # Mock invalid repo
            print("✅ scan_gitman_projects (mock): Handled invalid repos")
        
        print("\n=== Mock Debug Tests Passed! ===")