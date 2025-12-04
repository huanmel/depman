from typing import Optional
from datetime import datetime  # Add if not present (for res['datetime'])
import time
from functools import wraps
import click
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from depman import CONFIG_NAME, CACHE_GIT_REPOS, CACHE_CONFIGS   
import yaml  # For manual YAML loads
from git import Repo, GitCommandError
from gitman.models import Config
from itertools import chain




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

def find_uninstalled_configs(d: Dict[str, Any]) -> List[str]:
    """Recursively iterate over all key-value pairs in a nested dictionary."""
    configs = []
    for key, value in d.items():
        
        if isinstance(value, dict):
            if value.get('rev_installed') is None:
                configs.append(key)
            if 'deps' in value and value['deps']:
                configs.extend(find_uninstalled_configs(value['deps']))
        else:
            pass
    return configs


def find_git_root(start: str = ".") -> Optional[str]:
    """Find the nearest Git root."""
    start_path = Path(start).resolve()
    current = start_path
    while current != current.parent:
        if (current / ".git").exists():
            return str(current)
        current = current.parent
    return None

@timeit
def find_all_configs(root: Path, repos_in: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load all .gitman.yml configs under root, flatten deps into structure.
    Returns: {'configs': {config_path: str: {'project_root': Path, 'location': str, 'deps': List[Dict{'name': str, 'repo': str, 'rev': str, 'path': Path}]}}}
    """
    configs = {}
    repos = repos_in.get("repos", {})
    for config_path in root.glob(f"**/{CONFIG_NAME}"):
        project_root = config_path.parent
        try:
            with open(config_path) as f:
                content = yaml.safe_load(f)
            
            location = content.get("location")
            deps = {}
            deps_locked={}
            # Flatten requirements
                
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
                dep_path_rel=str(dep_path.relative_to(root))
                rev_installed = repos.get(dep_path_rel, {}).get("rev") if dep_path_rel in repos else None
                rev_locked=deps_locked[dep_path_rel]["rev"] if dep_path_rel in deps_locked else None
                rev = req["rev"]
                if rev_installed:
                    proj_root_relative=str(project_root.relative_to(root))
                    repos[dep_path_rel]["used_in_configs"].update({proj_root_relative : rev_locked if rev_locked else rev_installed})
                    
                    
                deps[dep_path_rel]={
                    "name": name,
                    "repo": req["repo"],
                    "rev": rev,
                    "rev_locked": rev_locked,
                    "rev_installed": rev_installed,
                    "path": dep_path_rel
                }
            project_root_short = str(project_root.relative_to(root))
            rev_installed = repos.get(project_root_short, {}).get("rev") if dep_path_rel in repos else None

            configs[str(project_root_short)] = {
                "project_root": project_root_short,
                "config_file": str(config_path.name),
                "location": location,
                "rev_installed": rev_installed,                
                "deps": deps
            }
        except Exception as e:
            click.echo(click.style(f"Error loading {config_path}: {e}", fg="red"))
    
    res = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configs": configs
    }
    repos_in["repos"] = repos
    return res, repos_in

@timeit
def find_all_git_repos(root: Path) -> Dict[str, Any]:
    """
    Find all Git repos under root, fetch upstream, and detect updates/uncommitted/unpushed.
    Handles detached HEAD: Uses origin/main (or origin/HEAD) for comparisons; flags behind/ahead.
    Returns: {
        'repos': {
            repo_path: str (relative): {
                'name': str (basename or "."),
                'project_root': str (relative path),
                'revision': str (full HEAD SHA),
                'short_revision': str (7 chars),
                'remote_url': str or None,
                'tags': List[str],
                'current_tag': str or None (matching on HEAD commit),
                'current_branch': str ("detached@<short_sha>" if detached),
                'has_update': bool (behind origin/main post-fetch),
                'update_details': Dict{'branch': str ('origin/main'), 'latest_hash': str, 'datetime': str, 'message': str} or None,
                'has_update_main': bool (redundant for detached; always checks main),
                'update_details_main': Dict[...] or None (same as update_details if behind),
                'has_uncommitted': bool,
                'uncommitted_files': List[str] (relative paths if dirty),
                'has_unpushed': bool (ahead of origin/main),
                'unpushed_count': int
            }
        },
        'datetime': str (scan time)
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
                
                # Handle current_branch and current_tag safely
                if repo.head.is_detached:
                    current_branch = f"detached@{short_revision}"
                    # Tag: Match on HEAD commit
                    matching_tags = [tag.name for tag in repo.tags if tag.commit.hexsha == revision]
                    current_tag = matching_tags[0] if matching_tags else None
                else:
                    current_branch = repo.active_branch.name
                    current_tag = (
                        repo.head.ref.name.replace("refs/tags/", "")
                        if repo.head.ref and repo.head.ref.name.startswith("refs/tags/")
                        else None
                    )

                # Fetch upstream (origin)
                if hasattr(repo.remotes, 'origin'):
                    repo.remotes.origin.fetch()

                # Determine remote baseline (origin/main or origin/HEAD)
                remote_ref_name = None
                remote_ref = None
                for ref_name in ['origin/main', 'origin/HEAD']:
                    try:
                        remote_ref = repo.refs[ref_name]
                        remote_ref_name = ref_name
                        break
                    except IndexError:
                        continue
                if not remote_ref:
                    current_branch = None  # Skip checks

                # Check updates (behind: HEAD..remote_ref) and unpushed (ahead: remote_ref..HEAD)
                has_update = False
                update_details = None
                has_unpushed = False
                unpushed_count = 0
                if remote_ref:
                    # Behind (updates available)
                    behind_commits = list(repo.iter_commits(f'HEAD..{remote_ref.name}'))
                    if behind_commits:
                        has_update = True
                        latest_commit = behind_commits[0]  # Most recent remote
                        update_details = {
                            "branch": remote_ref_name,
                            "latest_hash": latest_commit.hexsha,
                            "datetime": latest_commit.authored_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                            "message": latest_commit.message.split('\n')[0]
                        }
                    
                    # Ahead (unpushed/diverged)
                    ahead_commits = list(repo.iter_commits(f'{remote_ref.name}..HEAD'))
                    unpushed_count = len(ahead_commits)
                    has_unpushed = unpushed_count > 0
                
                
                # get latest commit info
                head_commit_info = None
                try:
                    commit = repo.head.commit
                    head_commit_info = {
                        "datetime": commit.authored_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        "message": commit.message.split('\n')[0],
                        "author": commit.author.name
                    }
                except (IndexError, AttributeError):
                    pass
                
                # Main check (always, if exists; for non-detached, as before)
                has_update_main = False
                update_details_main = None
                if not repo.head.is_detached:
                    if current_branch and current_branch != "main":
                        for main_branch in ["main", "master"]:
                            try:
                                main_ref_name = f"origin/{main_branch}"
                                origin_main_ref = repo.refs[main_ref_name]
                                main_local_ref = repo.refs[main_branch]
                                if main_local_ref.commit.hexsha != origin_main_ref.commit.hexsha:
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
                else:
                    # For detached, has_update_main mirrors has_update (main divergence)
                    has_update_main = has_update
                    update_details_main = update_details

                # Uncommitted changes (works in detached)
                has_uncommitted = repo.is_dirty()
                uncommitted_files = []
                if has_uncommitted:
                    uncommitted_files.extend(repo.untracked_files)
                    uncommitted_files.extend([diff.a_path for diff in repo.index.diff(None)])
                    uncommitted_files.extend([diff.a_path for diff in repo.index.diff("HEAD")])

                project_root_short = git_root.relative_to(root)
                repos[str(project_root_short)] = {
                    "name": project_root_short.name if project_root_short.name else ".",
                    "project_root": str(project_root_short),
                    "rev": revision,
                    "rev_short": short_revision,
                    "remote_url": remote_url,
                    "tags": tags,
                    "current_tag": current_tag,
                    "current_branch": current_branch,
                    "commit_info": head_commit_info,
                    "has_update": has_update,
                    "update_details": update_details,
                    "has_update_main": has_update_main,
                    "update_details_main": update_details_main,
                    "has_uncommitted": has_uncommitted,
                    "uncommitted_files": uncommitted_files,
                    "has_unpushed": has_unpushed,
                    "unpushed_count": unpushed_count,
                    "used_in_configs": {},
                }
            except Exception as e:
                click.echo(click.style(f"  {repo_path}: Error ({e})", fg="red"))
    res = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repos": repos
    }
    return res


"""
analyze_configs_repos: Cross-analyze Gitman configs and Git repos dicts.
Expands both in-place (mutates originals for simplicity; returns them for chaining).
Requires: GitPython (git) for precise 'behind_main' checks (compares rev_locked to origin/main SHA).
If dep path not in repos or no Git repo, sets behind_main=False.
"""

def print_list_configs_repos(
    configs: Dict[str, Any], repos: Dict[str, Any],only_dirty: bool=False
):
    """print configs and repos summary."""
    click.echo("\nRepos Summary:")
    print(f"Total repos: {len(repos.get('repos', {}))}")
    print(f"snapshot datetime: {repos.get('datetime', {})}")
    for repo_path, repo_info in repos.get("repos", {}).items():
        has_updates_repo = repo_info["has_uncommitted"] or repo_info["has_unpushed"] or repo_info["has_update"] or repo_info["has_update_main"]
        conf=repo_info.get('used_in_configs')
        is_rev_matched = False
        if conf:
            revs=list(conf.values())
            # check if all revs match the installed rev
            is_rev_matched = all( rev == repo_info.get('rev') for rev in revs)            
      
        msg_upd_style='bright_yellow' if has_updates_repo else 'bright_green'
        
        conf_style = 'bright_green' if is_rev_matched else 'bright_yellow'
        
        
        is_all_ok = not has_updates_repo and is_rev_matched
        
        if only_dirty and is_all_ok:
            continue
        
        sym =  "✅" if is_all_ok else "⚠️"
        
        
        click.echo(f"{sym}  repo: {repo_path}        branch rev: {repo_info.get('current_branch')} {repo_info.get('rev_short')}")
        click.echo(f"       datetime: {repo_info.get('commit_info')['datetime']} | message: {repo_info.get('commit_info')['message']}")
        if has_updates_repo:
            click.echo(click.style(f"         | Has Update: {repo_info.get('has_update')} | Uncommitted: {repo_info.get('has_uncommitted')} | Unpushed: {repo_info.get('has_unpushed')}",fg=msg_upd_style))
            if repo_info.get('has_update') and repo_info.get('update_details'):
                ud = repo_info.get('update_details')
                click.echo(f"         | Update Details: Branch: {ud.get('branch')} | Latest Hash: {ud.get('latest_hash')} | Datetime: {ud.get('datetime')} | Message: {ud.get('message')}")
        click.echo(click.style(f"         | configs: {repo_info.get('used_in_configs')}",fg=conf_style))
        click.echo()
        
        # list configs without installations
        # get all items wihtout rev_installed
        
    uninstalled_configs = find_uninstalled_configs(configs.get("configs", {}))
    if uninstalled_configs:
        click.echo("⚠️ Configs without installations:")
        for uc in uninstalled_configs:
            click.echo(click.style(f"  - {uc}", fg="red"))

    # click.echo("Configs Summary:")
    # for config_path, config_data in configs.get("configs", {}).items():
    #     click.echo(f"  Config: {config_path}")
    #     for dep_path, dep_info in config_data.get("deps", {}).items():
    #         click.echo(f"    Dep: {dep_path} | Rev Locked: {dep_info.get('rev_locked')} | In Repos: {dep_info.get('in_repos')} | Rev Match: {dep_info.get('rev_match')} | Behind Main: {dep_info.get('behind_main')}")
    

@timeit
def analyze_configs_repos(
    configs: Dict[str, Any], repos: Dict[str, Any], root: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze and expand configs/repos dicts with cross-references.
    
    Adds to configs['.']['deps'][dep_path]:
    - 'in_repos': bool (dep_path in repos)
    - 'rev_match': bool (rev_locked == repos[dep_path]['revision'] if in_repos)
    - 'behind_main': bool (rev_locked behind origin/main; uses GitPython fetch/compare)
    
    Adds to repos[repo_path]:
    - 'in_config': bool (repo_path in configs['.']['deps'])
    
    Args:
        configs: Loaded configs dict (e.g., from load_all_configs()['configs'])
        repos: Loaded repos dict (e.g., from find_all_git_repos()['repos'])
        root: Project root Path (for Git ops on deps)
    
    Returns:
        Tuple[configs, repos] (expanded originals)
    
    Example:
        configs, repos = analyze_configs_repos(configs, repos, Path('.'))
        # Now configs['.']['deps']['.deps\\mbd']['behind_main'] == True/False
    """
    # Normalize paths (Windows \ vs /; use str keys as-is)
    config_deps = configs.get('.', {}).get('deps', {})
    for dep_path_str, dep_info in config_deps.items():
        dep_path = Path(dep_path_str)
        in_repos = dep_path_str in repos
        dep_info['in_repos'] = in_repos
        
        rev_match = False
        behind_main = False
        if in_repos:
            repo_info = repos[dep_path_str]
            rev_match = dep_info['rev_locked'] == repo_info['rev']
            dep_info['rev_match'] = rev_match
            
            # Check if rev_locked behind origin/main (precise Git check)
            try:
                repo = Repo(root / dep_path)
                if hasattr(repo.remotes, 'origin'):
                    repo.remotes.origin.fetch()
                # Find origin/main or origin/HEAD
                remote_ref_name = None
                remote_ref = None
                for ref_name in ['origin/main', 'origin/HEAD']:
                    try:
                        remote_ref = repo.refs[ref_name]
                        remote_ref_name = ref_name
                        break
                    except IndexError:
                        continue
                if remote_ref and dep_info['rev_locked']:
                    # Check if rev_locked..remote_ref has commits (behind)
                    behind_commits = list(repo.iter_commits(f"{dep_info['rev_locked']}..{remote_ref.name}"))
                    behind_main = len(behind_commits) > 0
            except (GitCommandError, ValueError):  # No repo/rev
                pass
        else:
            dep_info['rev_match'] = False
        dep_info['behind_main'] = behind_main
    
    # For repos: Check if in config deps
    for repo_path_str, repo_info in repos['repos'].items():
        repo_info['in_config'] = repo_path_str in config_deps
    
    return configs, repos

@timeit
def find_all_git_repos1(root: Path) -> Dict[str, Any]:
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
                    "unpushed_count": unpushed_count,
                    "configs" : [],
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
    
def get_cashed_configs(root,CACHE_GIT_REPOS, CACHE_CONFIGS):
    loaded_configs = yaml.safe_load(open(root/ CACHE_CONFIGS))
    git_repos = yaml.safe_load(open( root / CACHE_GIT_REPOS))
    print(f"✅ Loaded cached configs from {str(root/ CACHE_CONFIGS)}") 
    print(f"✅ Loaded cached git status from {str(root/ CACHE_GIT_REPOS)}")
    return loaded_configs, git_repos

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

def find_nested_configs(base_path: Path, depth: int = float("inf")) -> List[Path]:
    """Find nested .gitman.yml files (simple walker; max depth to avoid cycles)."""
    configs = []
    for config_path in base_path.glob(f"**/{CONFIG_NAME}"):
        rel_depth = len(config_path.relative_to(base_path).parts) - 1
        if depth > 0 and rel_depth >= depth:
            continue
        configs.append(config_path)
    return configs



def get_configs_and_repos(root: Path, use_cache: bool = False):
    """Get configs and git repos, using cache if specified."""
    if use_cache:
        loaded_configs, git_repos = get_cashed_configs(
            root, CACHE_GIT_REPOS, CACHE_CONFIGS)

    else:
        git_repos = find_all_git_repos(root)
        num_repos = len(git_repos["repos"])
        print(f"✅ find_all_git_repos: Found {num_repos} repos")
        
        loaded_configs, git_repos = find_all_configs(root, git_repos)
        # num_configs = len(loaded_configs["configs"])
        cache_conf_file = root / CACHE_CONFIGS
        with open(cache_conf_file, "w") as f:
            yaml.safe_dump(loaded_configs, f,
                           default_flow_style=False, sort_keys=False)
        print(f"✅ Dumped loaded_configs to {cache_conf_file}")


        cache_repos_file = root / CACHE_GIT_REPOS
        with open(cache_repos_file, "w") as f:
            yaml.safe_dump(
                git_repos, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Dumped git_repos to {cache_repos_file}")
    return loaded_configs, git_repos