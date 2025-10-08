"""
Update checker using Gitman API + GitPython.
"""

import click
from pathlib import Path
from typing import List, Dict, Any
from git import Repo, GitCommandError
from gitman.models import Config


def find_nested_configs(base_path: Path, depth: int = float("inf")) -> List[Path]:
    """Find nested .gitman.yml files (simple walker; max depth to avoid cycles)."""
    configs = []
    for config_path in base_path.rglob(".gitman.yml"):
        if depth > 0 and config_path.relative_to(base_path).parts[:1].count("/") >= depth:
            continue  # Rough depth limit
        configs.append(config_path)
    return configs


def get_all_requirements(config: Config, recursive: bool = False) -> List[Dict[str, Any]]:
    """Flatten requirements + groups (no public get_dependencies, so manual)."""
    reqs = config.get_requirements()
    for group_name, group_reqs in config.groups.items():
        reqs.extend(group_reqs)
    if recursive:
        location = Path(config.directory or ".") / config.content.get("location", "requirements")
        for nested_config_path in find_nested_configs(location):
            nested_config = Config.load(nested_config_path)
            reqs.extend(get_all_requirements(nested_config, recursive=False))  # Avoid infinite recursion
    return reqs


@click.command()
@click.option("--recursive", is_flag=True, help="Check nested configs too.")
@click.pass_context
def check_updates_cmd(ctx: click.Context, recursive: bool):
    """Check for upstream updates in Gitman dependencies."""
    config = Config.load_config(root)
    depth=None
    allow_dirty=True
    if config and config.root:
        skip_paths = []
        for identity in config.get_dependencies(depth=depth, allow_dirty=allow_dirty):
            count += 1
            config.log("{}: {} @ {}", *identity)
            skip_paths.append(identity.path)

        nested_configs = find_nested_configs(config.root, depth, skip_paths)
        # if nested_configs:
        #      for nested_config in nested_configs:
        #         for identity in nested_config.get_dependencies(
        #             depth=depth, allow_dirty=allow_dirty
        #         ):
        
    root = ctx.obj["root"]
    config_path = root / "gitman.yml"
    if not config_path.exists():
        raise click.ClickException(f"No gitman.yml in {root}")

    config = Config.load(config_path)
    location = root / config.content.get("location", "requirements")
    all_reqs = get_all_requirements(config, recursive)

    if not all_reqs:
        click.echo("No requirements found.")
        return

    click.echo(f"Scanning {len(all_reqs)} requirements in {location}...")

    for req in all_reqs:
        name = req.get("name", Path(req["repo"]).name)
        repo_url = req["repo"]
        target_rev = req["rev"]
        dep_path = location / name

        if not dep_path.exists():
            click.echo(click.style(f"{name}: Not installed", fg="yellow"))
            continue

        try:
            repo = Repo(dep_path)
            repo.remotes.origin.fetch()

            current_sha = repo.head.commit.hexsha

            # Determine if branch (updatable) or tag/commit
            try:
                origin_ref = repo.refs[f"origin/{target_rev}"]
                latest_sha = origin_ref.commit.hexsha
                is_branch = True
            except IndexError:
                # Fall back to local rev_parse (for tags/commits)
                latest_sha = repo.rev_parse(target_rev).hexsha
                is_branch = False

            short_current = current_sha[:8]
            short_latest = latest_sha[:8]

            if current_sha == latest_sha:
                status = "Up to date"
                color = "green"
            else:
                if is_branch:
                    status = f"Update available! ({short_current} -> {short_latest} on {target_rev})"
                    color = "red"
                else:
                    status = f"Mismatch (pinned to {target_rev}: {short_latest})"
                    color = "yellow"

            click.echo(click.style(f"{name}: {status} | {repo_url}", fg=color))

        except GitCommandError as e:
            click.echo(click.style(f"{name}: Git error ({e})", fg="red"))
        except KeyError as e:
            click.echo(click.style(f"{name}: Missing key ({e})", fg="yellow"))