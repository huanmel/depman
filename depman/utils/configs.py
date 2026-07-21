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
from rich.console import Console
from rich.table import Table
import os

from gitman import update as gitman_update
from tqdm import tqdm
import pyperclip 




def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function: {func.__name__} \nTimeit: {total_time:.4f} sec')
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
            rev_installed = repos.get(project_root_short, {}).get("rev")

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
    paths=[p for p in root.rglob(".git") if p.is_dir()]
    for repo_path in tqdm(paths, desc="Scanning Git repos",colour="green"):
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


def print_check_table(
    configs: Dict[str, Any], git_repos: Dict[str, Any], root: Path=Path('.'), list_mode: bool=False, list_open_terminal: bool = True, update_mode: bool=False):
    # Table for updates
    table = Table(title="Git deps status", show_header=True,
                  header_style="bold magenta")
    cols = ["#", "Dep", "Status", "Uncommitted",
            "Unpushed", "Update", "Update Main"]
    for col in cols:
        table.add_column(col, style="dim", overflow="fold")

    has_updates = False
    updates_projects = []
    click.echo(click.style(
        f"\n=== Git Repos Status (scanned at {git_repos['datetime']}) ===", bold=True))

    for idx, (key, val) in enumerate(git_repos["repos"].items()):
        has_updates_repo = val["has_uncommitted"] or val["has_unpushed"] or val["has_update"] or val["has_update_main"]

        if val["has_uncommitted"]:
            click.echo(click.style(
                f"⚠️  Repo {key} has uncommitted changes: {val['uncommitted_files']}", fg="yellow"))
            has_updates_repo = True
        if val["has_unpushed"]:
            click.echo(click.style(
                f"⚠️  Repo {key} has {val['unpushed_count']} unpushed commits", fg="yellow"))
        if val["has_update"]:
            details = val["update_details"]
            click.echo(click.style(
                f"⚠️  Repo {key} has updates on {details['branch']}@{details['latest_hash'][:7]}:{details.get('datetime')} - {details['message']}", fg="yellow"))
            updates_projects.append(key)
        if val["has_update_main"]:
            details = val["update_details_main"]
            click.echo(click.style(
                f"⚠️  Repo {key} has updates on main branch {details['branch']}: {details['latest_hash'][:7]} - {details['message']}", fg="yellow"))

        has_updates = has_updates or has_updates_repo
        table.add_row(*[str(idx),
            key,
            "⚠️" if has_updates_repo else "✅",
            "⚠️" if val["has_uncommitted"] else "✅",
            "⚠️" if val["has_unpushed"] else "✅",
            "⚠️" if val["has_update"] else "✅",
            "⚠️" if val["has_update_main"] else "✅"
        ],  style='bright_green' if not has_updates_repo else 'bright_yellow')
        # table.add_row(
        #     key,
        #     "!" if has_updates_repo else "ok",
        #     "!" if val["has_uncommitted"] else "ok",
        #     "!" if val["has_unpushed"] else "ok",
        #     "!" if val["has_update"] else "ok",
        #     "!" if val["has_update_main"] else "ok",
        #     style='bright_green' if not has_updates_repo else 'bright_yellow'
        # )

    console = Console()
    console.print(table)
    if list_mode:
        click.echo(click.style(
            "select #:", fg="green"))
        # c = click.getchar(echo=True)
        #     click.echo("Enter a string: ")
        # Standard Python input() reads a full line as a string
        c = input()
        click.echo(f"You entered: {c}")
        click.echo()
        if c.isdigit():
            idx_sel = int(c)
            if idx_sel >=0 and idx_sel < len(git_repos["repos"]):
                repo_key = list(git_repos["repos"].keys())[idx_sel]
                click.echo(click.style(
                    f"Details for repo {c} {repo_key}:", fg="green"))
                repo_info = git_repos["repos"][repo_key]
                click.echo(yaml.dump(repo_info, sort_keys=False))
                project_root = Path(root) / repo_key
                command = f'cd {project_root} && git status'
                click.echo(click.style(
                    f"cd {project_root}", fg="cyan"))
                # os.chdir(project_root)
                # os.system(f'cd {project_root}')
                pyperclip.copy(command)
                
                if list_open_terminal:
                    # wt command with various options:
                    # -w 0: Use current window (0 means "always open here")
                    # split-pane -V: open a new vertical split pane
                    # -d: Set starting directory
                    wt_command = [
                        'wt.exe',
                        '-w', '0',
                        'split-pane',
                        '-V',
                        '-d', str(project_root)
                    ]
                    print(f"\nTo open in Windows Terminal, run:")
                    print(" ".join(wt_command))

                    # Try to execute it
                    try:
                        import subprocess
                        subprocess.run(wt_command, shell=False)
                    except Exception:
                        print("Make sure Windows Terminal is installed")
            else:
                click.echo(click.style(
                    f"Invalid selection: {c}", fg="red"))
        return
    
    if update_mode and has_updates:
        click.echo(click.style(
            "⚠️  Updates available—run 'depman gm update' to apply.", fg="yellow"))
        click.echo("\n".join(updates_projects))
        click.echo('apply updates? [yn] ', nl=False)
        c = click.getchar()
        click.echo()
        if c == 'y':
            click.echo('apply updates')
            # gitman_update(*updates_projects,root=root)
            # gitman_update(root=root)
            for p in updates_projects:
                click.echo(click.style(f"apply update for {p}? [y/enter/n] ", fg="yellow"))
                print("Current commit info:")
                print(git_repos['repos'][p]['commit_info']['datetime'], "|", git_repos['repos'][p]['commit_info']['message'])
                print("Update details:")
                print(git_repos['repos'][p]['update_details']['datetime'], "|", git_repos['repos'][p]['update_details']['message'])
                c = click.getchar()
                click.echo()
                if (c == 'y' or c == '\n'):
                    full_p=Path(root,p)
                    conf_name = find_dep_config_name(configs, p)
                    if conf_name:
                        print(f'Updating {p} in config {conf_name}')
                        gitman_update(conf_name,root=root)
                        click.echo(click.style(f"Updated {p}", fg="green"))
                    else:
                        click.echo(click.style(f"Skipped {p} (not in any config)", fg="red"))
                else:
                    click.echo(f'Skipped {p}')
        elif c == 'n':
            click.echo('Abort!')
        else:
            click.echo('Invalid input :(')


def find_dep_config_name(configs: Dict[str, Any], dep_path: str) -> Optional[str]:
    """Find the dep name for dep_path by searching every loaded config (root and nested), not just root."""
    for config_entry in configs.get('configs', {}).values():
        deps = config_entry.get('deps', {})
        if dep_path in deps:
            return deps[dep_path]['name']
    return None


def print_list_configs_repos(
    configs: Dict[str, Any], repos: Dict[str, Any],only_dirty: bool=False, list_mode = False
):
    """print configs and repos summary."""
    click.echo("\nRepos Summary:")
    print(f"Total repos: {len(repos.get('repos', {}))}")
    print(f"snapshot datetime: {repos.get('datetime', {})}")
    is_total_ok = True
    for repo_path, repo_info in repos.get("repos", {}).items():
        has_updates_repo = repo_info["has_uncommitted"] or repo_info["has_unpushed"] or repo_info["has_update"] or repo_info["has_update_main"]
        conf=repo_info.get('used_in_configs')
        is_rev_matched = False
        if repo_path == ".":
            # skip root repo
            is_rev_matched = True
        else:
            if conf:
                revs=list(conf.values())
                # check if all revs match the installed rev
                is_rev_matched = all( rev == repo_info.get('rev') for rev in revs)            
      
        msg_upd_style='bright_yellow' if has_updates_repo else 'bright_green'
        
        conf_style = 'white' if is_rev_matched else 'bright_yellow'
        
        
        is_all_ok = not has_updates_repo and is_rev_matched
        if not is_all_ok:
            is_total_ok = False
        
        if only_dirty and is_all_ok:
            continue
        
        sym =  "✅" if is_all_ok else "⚠️"
        
        
        click.echo(f"{sym}  repo: {repo_path}        branch@rev: {repo_info.get('current_branch')}@{repo_info.get('rev_short')}")
        click.echo(f"       datetime: {repo_info.get('commit_info')['datetime']} | message: {repo_info.get('commit_info')['message']}")
        if has_updates_repo:
            if repo_info.get('has_uncommitted'):
                click.echo(click.style(f"         | Has Uncommitted Changes: {repo_info.get('has_uncommitted')} | Files: {', '.join(repo_info.get('uncommitted_files', []))}",fg='bright_yellow'))
            if repo_info.get('has_unpushed'):
                click.echo(click.style(f"         | Has Unpushed Commits: {repo_info.get('has_unpushed')} | Count: {repo_info.get('unpushed_count')}",fg='bright_yellow'))
            if repo_info.get('has_update'):
                click.echo(click.style(f"         | Update Available: {repo_info.get('has_update')}",fg='yellow'))
            # click.echo(click.style(f"         | Has Update: {repo_info.get('has_update')} | Uncommitted: {repo_info.get('has_uncommitted')} | Unpushed: {repo_info.get('has_unpushed')}",fg=msg_upd_style))
            if repo_info.get('has_update') and repo_info.get('update_details'):
                ud = repo_info.get('update_details')
                click.echo(f"         | Update Details: Branch: {ud.get('branch')} | Latest Hash: {ud.get('latest_hash')[:7]} | Datetime: {ud.get('datetime')} | Message: {ud.get('message')}")
        if (not is_rev_matched or not only_dirty):
            click.echo(click.style(f"       configs: {repo_info.get('used_in_configs')}",fg=conf_style))
            click.echo()

    uninstalled_configs = find_uninstalled_configs(configs.get("configs", {}))
    if uninstalled_configs:
        click.echo("⚠️ Configs without installations:")
        for uc in uninstalled_configs:
            click.echo(click.style(f"  - {uc}", fg="yellow"))
    is_total_ok = is_total_ok and (len(uninstalled_configs) == 0)

    behind_main_deps = [
        f"{dep_path} (in config {config_path})"
        for config_path, config_entry in configs.get("configs", {}).items()
        for dep_path, dep_info in config_entry.get("deps", {}).items()
        if dep_info.get("behind_main")
    ]
    if behind_main_deps:
        click.echo("⚠️ Deps whose locked revision is behind origin/main:")
        for d in behind_main_deps:
            click.echo(click.style(f"  - {d}", fg="yellow"))
    is_total_ok = is_total_ok and (len(behind_main_deps) == 0)

    if is_total_ok:
        click.echo(click.style("✅ All repos are up-to-date, installed revisions matching configs.", fg="green"))


@timeit
def analyze_configs_repos(
    configs: Dict[str, Any], repos: Dict[str, Any], root: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze and expand configs/repos dicts with cross-references.
    Covers every loaded config (root and nested), not just the root config.

    Adds to configs['configs'][config_path]['deps'][dep_path]:
    - 'in_repos': bool (dep_path in repos['repos'])
    - 'rev_match': bool (rev_locked == repos['repos'][dep_path]['rev'] if in_repos)
    - 'behind_main': bool (rev_locked behind origin/main; uses GitPython fetch/compare)

    Adds to repos['repos'][repo_path]:
    - 'in_config': bool (repo_path declared as a dep in any loaded config)

    Args:
        configs: Loaded configs dict, the {'configs': {...}} shape from find_all_configs()
        repos: Loaded repos dict, the {'repos': {...}} shape from find_all_git_repos()
        root: Project root Path (for Git ops on deps)

    Returns:
        Tuple[configs, repos] (expanded originals)
    """
    repos_map = repos.get('repos', {})
    all_dep_paths = set()
    for config_entry in configs.get('configs', {}).values():
        for dep_path_str, dep_info in config_entry.get('deps', {}).items():
            all_dep_paths.add(dep_path_str)
            in_repos = dep_path_str in repos_map
            dep_info['in_repos'] = in_repos

            rev_match = False
            behind_main = False
            if in_repos:
                repo_info = repos_map[dep_path_str]
                rev_match = dep_info['rev_locked'] == repo_info['rev']

                # Check if rev_locked behind origin/main (precise Git check)
                try:
                    repo = Repo(root / dep_path_str)
                    if hasattr(repo.remotes, 'origin'):
                        repo.remotes.origin.fetch()
                    # Find origin/main or origin/HEAD
                    remote_ref = None
                    for ref_name in ['origin/main', 'origin/HEAD']:
                        try:
                            remote_ref = repo.refs[ref_name]
                            break
                        except IndexError:
                            continue
                    if remote_ref and dep_info['rev_locked']:
                        # Check if rev_locked..remote_ref has commits (behind)
                        behind_commits = list(repo.iter_commits(f"{dep_info['rev_locked']}..{remote_ref.name}"))
                        behind_main = len(behind_commits) > 0
                except (GitCommandError, ValueError):  # No repo/rev
                    pass
            dep_info['rev_match'] = rev_match
            dep_info['behind_main'] = behind_main

    # For repos: Check if declared as a dep in any config
    for repo_path_str, repo_info in repos_map.items():
        repo_info['in_config'] = repo_path_str in all_dep_paths

    return configs, repos

def is_git_repo(path: Path) -> bool:
    """Check if path is a Git repo."""
    try:
        Repo(str(path), search_parent_directories=False)
        return True
    except Exception:
        return False
    
def get_cached_configs(root, CACHE_GIT_REPOS, CACHE_CONFIGS):
    with open(root / CACHE_CONFIGS) as f:
        loaded_configs = yaml.safe_load(f)
    with open(root / CACHE_GIT_REPOS) as f:
        git_repos = yaml.safe_load(f)
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
        loaded_configs, git_repos = get_cached_configs(
            root, CACHE_GIT_REPOS, CACHE_CONFIGS)

    else:
        git_repos = find_all_git_repos(root)
        num_repos = len(git_repos["repos"])
        print(f"✅ find_all_git_repos: Found {num_repos} repos")
        
        loaded_configs, git_repos = find_all_configs(root, git_repos)
        loaded_configs, git_repos = analyze_configs_repos(loaded_configs, git_repos, root)
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