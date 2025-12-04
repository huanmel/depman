"""
Update checker using Gitman API + GitPython.
"""

import click
from pathlib import Path
from git import Repo, GitCommandError
from gitman.models import Config
import yaml  # For manual YAML loads
from rich.console import Console
from rich.table import Table
from depman.utils.configs import  get_configs_and_repos, print_list_configs_repos
from depman import CONFIG_NAME, CACHE_GIT_REPOS, CACHE_CONFIGS


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
# @click.option("--recursive", is_flag=True, help="Flatten deps recursively for updates (legacy).")
# @click.option("--scan-depth", type=int, default=None, help="Max recursion depth for scanning (default: unlimited).")
@click.option("--use-cache",  is_flag=True, default=False, help="use cached YAML files instead of live scan (path to root).")
@click.pass_context
# , recursive: bool, scan_depth: Optional[int]
def check_updates_cmd(ctx: click.Context, use_cache: bool):
    """Check for upstream updates in Gitman dependencies (now scans all projects)."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache)

    # Table for updates
    table = Table(title="Git deps status", show_header=True,
                  header_style="bold magenta")
    cols = ["Dep", "Status", "Uncommitted",
            "Unpushed", "Update", "Update Main"]
    for col in cols:
        table.add_column(col, style="dim", overflow="fold")

    has_updates = False
    click.echo(click.style(
        f"\n=== Git Repos Status (scanned at {git_repos['datetime']}) ===", bold=True))

    for key, val in git_repos["repos"].items():
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
                f"⚠️  Repo {key} has updates on branch {details['branch']}: {details['latest_hash'][:7]} - {details['message']}", fg="yellow"))
        if val["has_update_main"]:
            details = val["update_details_main"]
            click.echo(click.style(
                f"⚠️  Repo {key} has updates on main branch {details['branch']}: {details['latest_hash'][:7]} - {details['message']}", fg="yellow"))

        has_updates = has_updates or has_updates_repo
        table.add_row(*[
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
    if has_updates:
        click.echo(click.style(
            "⚠️  Updates available—run 'depman gm update' to apply.", fg="yellow"))
    return  # Early return to avoid full project scan for now
    for project_root, config in projects:
        click.echo(click.style(
            f"\n=== Project: {project_root} ===", bold=True))
        print_project_tree(config, project_root)

        location = project_root / \
            config.content.get("location", "requirements")
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
                        status = click.style(
                            f"Update avail. ({short_current} → {short_latest})", fg="red")
                    else:
                        status = click.style(
                            f"Mismatch ({short_latest})", fg="yellow")

                table.add_row([name, status, repo_url])

            except GitCommandError as e:
                status = click.style(f"Git error ({e})", fg="red")
                table.add_row([name, status, repo_url])
            except KeyError as e:
                status = click.style(f"Missing key ({e})", fg="yellow")
                table.add_row([name, status, repo_url])

        click.echo(table)
        if has_updates:
            click.echo(click.style(
                "⚠️  Updates available—run 'depman gm update' to apply.", fg="yellow"))


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--use-cache",  is_flag=True, default=False, help="use cached YAML files instead of live scan (path to root).")
@click.option("--dirty",  is_flag=True, default=False, help="print only dirty.")
@click.pass_context
def list_cmd(ctx: click.Context, use_cache: bool, dirty: bool):
    """List all Git/Gitman projects under root."""
    root = ctx.obj["root"]
    loaded_configs, git_repos = get_configs_and_repos(root, use_cache=use_cache)
    print_list_configs_repos(loaded_configs, git_repos, only_dirty=dirty)
    
    


if __name__ == "__main__":
    from depman.cli import cli
    root = r'C:\Users\ivanm\Documents\MATLAB\EKL\deps_tests\gitman_proj_test'
    # cli(['--root', root, 'check-updates'])
    cli(['--root', root, 'list', '--use-cache','--dirty'])
    

    # import tempfile
    # import argparse  # Simple arg parse for path (lightweight)

    # parser = argparse.ArgumentParser(description="Debug Checker Functions")
    # parser.add_argument("--test-path", type=str, default=None, help="Path to real Gitman project for full test (overrides mocks)")
    # parser.add_argument("--use-cache", type=str, default=None, help="Path to root with cached YAML files (overrides live scan)")
    # args = parser.parse_args()

    # print("=== Debug Tests for Checker Functions ===")
    # if args.test_path:
    #     # Full real-project test
    #     root = Path(args.test_path).resolve()
    #     if not root.exists():
    #         raise ValueError(f"Test path {root} does not exist.")
    #     print(f"🔍 Testing with real project: {root}")
    #     if args.use_cache:
    #         loaded_configs, git_repos = get_cashed_configs(root, CACHE_GIT_REPOS, CACHE_CONFIGS)
    #     else:
    #         loaded_configs = find_all_configs(root)

    #     # New: Test load_all_configs

    #         cache_conf_file=root / CACHE_CONFIGS
    #         with open(cache_conf_file, "w") as f:
    #             yaml.safe_dump(loaded_configs, f, default_flow_style=False, sort_keys=False)
    #         print(f"✅ Dumped loaded_configs to {cache_conf_file}")

    #         assert "configs" in loaded_configs, "No 'configs' key"
    #         num_configs = len(loaded_configs["configs"])
    #         assert num_configs > 0, f"No configs found in {root}"
    #         print(f"✅ load_all_configs: Loaded {num_configs} configs")
    #         print("Sample config structure:", list(loaded_configs["configs"].keys())[:1])  # First key
    #         if num_configs > 0:
    #             sample_deps = loaded_configs["configs"]['.']["deps"]
    #             print(f"  Sample deps: {len(sample_deps)} (e.g., {next(iter(sample_deps.values())) if sample_deps else 'none'})")

    #         # New: Test find_all_git_repos
    #         git_repos = find_all_git_repos(root)
    #         assert "repos" in git_repos, "No 'repos' key"
    #         num_repos = len(git_repos["repos"])
    #         print(f"✅ find_all_git_repos: Found {num_repos} repos")
    #         if num_repos > 0:
    #             sample_repo = list(git_repos["repos"].values())[0]
    #             print(f"  Sample repo info: rev={sample_repo['revision'][:8]}, remote={sample_repo['remote_url']}, tags={len(sample_repo['tags'])}")

    #         cache_repos_file=root / CACHE_GIT_REPOS
    #         with open(cache_repos_file, "w") as f:
    #             yaml.safe_dump(git_repos, f, default_flow_style=False, sort_keys=False)
    #         print(f"✅ Dumped git_repos to {cache_repos_file}")

    #     configs, repos = analyze_configs_repos(loaded_configs, git_repos, root)
    #     cache_conf_file=root / CACHE_CONFIGS
    #     cache_repos_file=root / CACHE_GIT_REPOS
    #     with open(cache_conf_file, "w") as f:
    #             yaml.safe_dump(loaded_configs, f, default_flow_style=False, sort_keys=False)
    #     with open(cache_repos_file, "w") as f:
    #             yaml.safe_dump(git_repos, f, default_flow_style=False, sort_keys=False)

    #     sys.exit(0)

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

    #     print("\n=== Real Project Debug Tests Passed! ===")
    # else:
    #     # Fallback mock tests (updated to use new functions)
    #     print("🧪 Running mock tests (use --test-path for real project)")

    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         tmp_path = Path(tmpdir)
    #         # Create nested .gitman.yml
    #         (tmp_path / "proj1" / CONFIG_NAME).write_text(yaml.dump({
    #             "requirements": [{"name": "dep1", "repo": "https://ex/dep1", "rev": "main"}]
    #         }))
    #         (tmp_path / "proj1" / "sub" / CONFIG_NAME).write_text(yaml.dump({
    #             "groups": {"g1": [{"name": "dep2", "repo": "https://ex/dep2", "rev": "v1"}]}
    #         }))
    #         loaded = find_all_configs(tmp_path)
    #         assert len(loaded["configs"]) == 2
    #         print(f"✅ load_all_configs (mock): {len(loaded['configs'])} configs")

    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         tmp_path = Path(tmpdir)
    #         proj1 = tmp_path / "proj1"
    #         proj1.mkdir()
    #         (proj1 / ".git").mkdir()  # Simple mock; full Repo needs init/commit in real
    #         assert is_git_repo(proj1) is False  # Empty .git not valid
    #         print("✅ is_git_repo: Handled invalid (for mock)")

    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         tmp_path = Path(tmpdir)
    #         config_path = tmp_path / CONFIG_NAME
    #         config_path.write_text(yaml.dump({
    #             "requirements": [{"name": "req1", "repo": "https://ex/req1", "rev": "main"}],
    #             "groups": {"testgroup": [{"name": "gdep1", "repo": "https://ex/gdep1", "rev": "develop"}]}
    #         }))
    #         (tmp_path / ".git").mkdir()
    #         config = Config.load(config_path)
    #         print("\n--- Sample Tree Output ---")
    #         print_project_tree(config, tmp_path)
    #         all_reqs = get_all_requirements(config)
    #         assert len(all_reqs) == 2
    #         print(f"✅ print_project_tree & get_all_requirements (mock): {len(all_reqs)} reqs")

    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         tmp_path = Path(tmpdir)
    #         proj_dir = tmp_path / "testproj"
    #         proj_dir.mkdir()
    #         (proj_dir / ".git").mkdir()
    #         (proj_dir / CONFIG_NAME).write_text(yaml.dump({
    #             "requirements": [{"name": "testdep", "repo": "https://ex/test", "rev": "main"}]
    #         }))
    #         projects = scan_gitman_projects(tmp_path)
    #         assert len(projects) == 0  # Mock invalid repo
    #         print("✅ scan_gitman_projects (mock): Handled invalid repos")

    #     print("\n=== Mock Debug Tests Passed! ===")
