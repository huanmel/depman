# Depman

CLI tool for managing and auditing [Gitman](https://github.com/jacebrowning/gitman) dependency trees. Wraps Gitman commands and adds a rich status dashboard across all nested git repos.


## Features

- **Recursive repo scan** — discovers all `.git` repos and `gitman.yml` configs under the project root, including deeply nested sub-dependencies
- **Rich status table** — color-coded dashboard showing uncommitted changes, unpushed commits, available updates on current branch, and updates on `main`
- **Interactive list mode** (`-l`) — select a repo by index to inspect its full Git status and commit details
- **Terminal mode** (`-t`) — opens the selected repo in a new Windows Terminal split pane (Windows only)
- **Interactive update mode** (`-u`) — per-project confirmation prompts before applying `gitman update`
- **YAML caching** (`-c`) — reuse a previous scan snapshot to skip slow network fetches
- **Gitman wrapper** (`depman gm`) — short aliases for `gitman install`, `update`, `list`, `init`
- **Detached HEAD support** — correctly handles repos pinned to a specific commit or tag

## Installation

```bash
git clone https://github.com/yourusername/depman.git
cd depman
pip install -e .
```

Requires Python 3.10+. Dependencies are managed via `pyproject.toml`: `click`, `gitman`, `gitpython`, `tqdm`, `rich`.

## Project structure

```
depman/
├── pyproject.toml
├── depman/
│   ├── cli.py               # CLI entry point (click groups)
│   ├── commands/
│   │   ├── checker.py       # check and list commands + display logic
│   │   └── gm_commands.py   # gitman API wrappers
│   └── utils/
│       ├── __init__.py
│       └── configs.py       # repo/config scanning and analysis
└── tests/
    └── test_depman.py       # pytest suite (scanning + CLI smoke tests)
```

## Usage

Run from any directory inside a Git project:

```bash
depman check               # scan all repos, show status table
depman check -l            # interactive: select repo by # to inspect
depman check -l -t         # same + open selected in Windows Terminal
depman check -u            # interactive update mode
depman check -c            # use cached scan (fast, no network)
depman list                # text summary of all repos and configs
depman list -d             # show only dirty repos
depman gm install <name>   # gitman install wrapper
depman gm update <name>    # gitman update wrapper
depman gm lock <name>      # gitman lock wrapper
depman gm uninstall        # gitman uninstall wrapper
depman --root /path check  # override auto-detected git root
```

### Status table columns

| Column | Meaning |
|---|---|
| Status | Overall health (⚠️ any issue, ✅ clean) |
| Uncommitted | Staged, unstaged, or untracked changes |
| Unpushed | Local commits not yet pushed to origin |
| Update | New commits available on the tracked remote branch |
| Update Main | `main`/`master` branch has diverged from `origin/main` |

### Interactive list mode

After the table prints, enter a repo number to inspect it:
- Prints full YAML details (branch, revision, commit message, update info)
- Copies `cd <path> && git status` to clipboard
- With `-t`: opens a new split pane in Windows Terminal at that path

## Limitations

- **Terminal mode is Windows-only** — uses `wt.exe`; no Linux/macOS equivalent implemented yet
- Caching writes `.cache_git_repos.yaml` and `.cache_configs.yaml` to the project root (ignored by git)
- Parallel repo fetching not yet implemented (scanning is sequential, slow on large trees)

## TODO / Future work

- [ ] Cross-platform terminal support (Linux: `gnome-terminal`/`konsole`, macOS: `Terminal.app`/`iTerm2`)
- [ ] Parallel repo fetching (currently sequential, slow on large trees)

## Recently fixed

- Update mode now searches all loaded configs (root and nested) for a dep's owning config, instead of only the root `.` config
- `analyze_configs_repos` is wired into the `check`/`list` flow; `rev_match`/`behind_main`/`in_repos` are computed for every config dep, and deps behind `origin/main` are reported in `list` output
- `pyperclip` and `pyyaml` added to `pyproject.toml` dependencies (previously missing, breaking a fresh install)
- Removed dead code: `find_all_git_repos1`, `scan_gitman_projects`, `depman/core/` (unused empty package)
- Real pytest suite covering `find_all_git_repos`, `find_all_configs`, and CLI `check`/`list` invocation
- `depman gm lock` and `depman gm uninstall` wrappers added

## Development

```bash
pip install -e ".[dev]"
black . && flake8
pytest
```

MIT License. Contributions welcome.
