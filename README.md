# Depman

CLI tool for managing and auditing [Gitman](https://github.com/jacebrowning/gitman) dependency trees. Wraps Gitman commands and adds a rich status dashboard across all nested git repos.


## Features

- **Recursive repo scan** — discovers all `.git` repos and `gitman.yml` configs under the project root, including deeply nested sub-dependencies
- **Rich status table** — color-coded dashboard showing uncommitted changes, unpushed commits, available updates on current branch, and updates on `main`
- **Interactive list mode** (`-l`) — select a repo by index to inspect its full Git status and commit details
- **Terminal mode** (`-t`) — opens the selected repo in a new Windows Terminal split pane (Windows only)
- **Interactive update mode** (`-u`) — per-project confirmation prompts before applying `gitman update`
- **YAML caching** (`-c`) — reuse a previous scan snapshot to skip slow network fetches
- **Gitman wrapper** (`depman gm`) — short aliases for `gitman install`, `update`, `list`, `init`, `lock`, `uninstall`
- **Detached HEAD support** — correctly handles repos pinned to a specific commit or tag
- **Semi-automatic review mode** (`depman review`) — walks every dirty repo one at a time, shows the diff, and lets you commit, revert (safely, via stash), or skip

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
depman review              # semi-auto commit/revert walkthrough of every dirty repo
depman review -c           # same, but reuse the last check/list scan snapshot (no rescan/fetch)
depman check -r            # full status table first, then straight into review (reuses this scan)
depman review --order root-first  # process root repo first instead of last
depman check --json        # machine-readable status: single JSON document on stdout
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

### `depman review` — semi-automatic commit/revert

Scans for every repo (root and nested deps) that needs attention — either it
has uncommitted changes, or its working tree is clean but it has commits not
yet pushed to `origin` — and walks through them one at a time (nested deps
before the root repo by default — see "Processing order" below):

1. Prints a compact numbered list of changed files (`git status --porcelain` style) for the repo — not the full diff, to avoid dumping a wall of text for repos with lots of changes. If the working tree is clean (the repo only made the list because of unpushed commits), it says so instead of an empty list, and the default action switches to **push**.
2. Prompts `commit / revert / diff / push / skip / quit`:
   - **commit** — re-prints the file list (highlighting how many are new/untracked) and asks "Commit all N file(s)?" before doing anything; only on confirmation does it stage everything (`git add -A`) and commit with a message you supply (or a default). Right afterward it goes into the **push** step below (so you're asked about pushing once, with the new commit already included in the preview). A no-op ("Nothing to commit.") if the working tree is already clean.
   - **revert** — asks for a second confirmation, then runs `git stash push -u -m "depman-revert-safety-<timestamp>"`. This is a **safety-net revert, not a hard delete**: it resets the working tree to `HEAD` but the changes remain fully recoverable with `git stash pop` (the recovery command is printed after every revert). A no-op if the working tree is already clean.
   - **diff** — asks which file # to show the diff for (blank = all files), prints it, then re-shows the file list and re-prompts for the same repo so you can look before deciding. A no-op if there are no changed files.
   - **push** — fetches, then prints exactly which local commits `origin/<branch>` doesn't have yet (short hash + first line of message) before asking "Push?" — so you always see what you're about to send before confirming. Independent of commit — useful for pushing commits made before this review session, or right after committing. Refuses (with a clear message, no crash) if there's no `origin` remote, the repo is in detached HEAD, or there's simply nothing to push. If the working tree still has changes to decide on, it loops back to the same repo after pushing; otherwise it moves on.
   - **skip** — leaves the repo untouched and moves to the next one.
   - **quit** — stops the review immediately.

Because revert always goes through a labeled stash instead of `git reset --hard`/`git clean`, and commit always shows the full file list (new files included) before asking for confirmation, no uncommitted work is ever silently discarded or swept into a commit by this command. Pushing always asks first too, since it's the one action here that touches something outside your machine.

Pass `-c`/`--use-cache` to skip the rescan (including the per-repo `fetch`) and reuse the snapshot written by the last `check`/`list` run — useful if you already saw the status and just want to act on it without waiting again. The list of *which* repos are dirty comes from that cached snapshot, but the diff shown and the commit/revert action itself always operate on the live working tree, so nothing stale is ever committed or stashed. If the cache is old, a repo that's changed since may not show up (or may show a diff different from what the cache implied) — rerun without `-c` if in doubt.

**`depman check -r`** runs the normal full status table first — so you see the whole project's state in one shot, the same overview `check -l -t` gives you — and then drops straight into the same review workflow, reusing that same scan instead of scanning twice.

**Processing order** (`--order`, available on both `review` and `check -r`):

- `root-last` (default) — nested dependency repos first, the root project repo last. Useful when the root's own state (e.g. `gitman.yml`, lockfiles) should be handled only after its dependencies have settled.
- `root-first` — alphabetical order, root repo (`.`) first.

## Automation (scripts / LLM agents)

`depman check --json` and `depman list --json` print a single JSON document to stdout instead of the human-oriented table/text output — nothing else is written to stdout in this mode, so it's safe to pipe (e.g. `depman check --json | jq .`). All diagnostic/progress output (the repo-scan progress bar, "✅ Dumped ..." cache messages, per-function timing) goes to stderr in every mode, not just `--json`, so redirecting stderr separately (`2>err.log`) always gets you clean data on stdout.

Shape: `{"configs": {...}, "repos": {...}}`.

- `configs.configs[project_path]` — one entry per loaded `gitman.yml` (`project_path` is `.` for the root, or a relative path like `deps/widget` for nested configs): `location`, `config_file`, `rev_installed`, and `deps[dep_path]` with `name`, `repo`, `rev`, `rev_locked`, `rev_installed`, `in_repos`, `rev_match`, `behind_main` (the last three come from cross-referencing the dep against its actual git repo — see `analyze_configs_repos` in `depman/utils/configs.py`).
- `repos.repos[repo_path]` — one entry per discovered git repo (`repo_path` relative to `--root`, `.` for the root repo itself): `name`, `rev`, `rev_short`, `current_branch`, `remote_url`, `tags`, `commit_info`, `has_uncommitted`, `uncommitted_files`, `has_unpushed`, `unpushed_count`, `has_update`/`update_details`, `has_update_main`/`update_details_main`, `used_in_configs`, `in_config`.

This is read-only status reporting — there's currently no non-interactive equivalent of `depman review`'s commit/revert/push actions (that's fully interactive today); scripting those still means driving `review`'s prompts directly.

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
