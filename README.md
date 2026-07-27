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
- **Semi-automatic review mode** (`depman review`) — walks every dirty repo one at a time, shows the diff, and lets you commit, revert (safely, via stash), push, or skip; optionally squashes not-yet-pushed commits into one before pushing; suggests relocking a subproject's `gitman.yml` when it's drifted out of sync with what's actually installed
- **Installable pre-push warning hook** (`depman hooks install`) — a normal `git push` warns (never blocks) if any of the pushed repo's gitman-declared dependencies have local uncommitted or unpushed issues

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
│   │   ├── review.py        # semi-automatic commit/revert/push review
│   │   ├── hooks.py         # installable pre-push warning hook
│   │   └── gm_commands.py   # gitman API wrappers
│   ├── backends/            # pluggable "where do declared/locked revisions come from"
│   │   ├── base.py          # ConfigBackend interface
│   │   └── gitman_backend.py  # the only implementation today
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
depman review --order gitman      # process deps in gitman.yml declared order (reorder it to control this)
depman check --json        # machine-readable status: single JSON document on stdout
depman check -j 16         # scan/fetch up to 16 repos concurrently (default: 8)
depman hooks install       # install a pre-push warning hook into this repo
depman hooks check         # preview what the hook would say, without installing it
depman hooks uninstall     # remove it (only if depman installed it)
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

Row order is deterministic: root (`.`) first, then each dep in the order it's
declared in its owning `gitman.yml` — depth-first, so a dep that itself owns a
nested `gitman.yml` is immediately followed by its own declared sub-deps,
rather than having them appear at the end. Reorder a `gitman.yml`'s `sources`
list to control this order (e.g. move a dep you want to handle first to the
top). Repos found on disk but not declared in any config (a "did I forget to
add this to gitman.yml?" case) are appended after, sorted alphabetically. This
same ordering also drives `--order gitman` in `review`/`check -r` below, and
list-mode's `select #:` index always matches the row numbers shown.

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
   - **push** — fetches, then prints exactly which local commits `origin/<branch>` doesn't have yet (short hash + first line of message) before asking "Push?" — so you always see what you're about to send before confirming. Independent of commit — useful for pushing commits made before this review session, or right after committing. Refuses (with a clear message, no crash) if there's no `origin` remote, the repo is in detached HEAD, or there's simply nothing to push. If the working tree still has changes to decide on, it loops back to the same repo after pushing; otherwise it moves on. **If there's more than one commit to push**, it first asks whether to squash them into one: shows the combined multiline message (each squashed commit's message, oldest first) and asks "Squash N commits into one?"; on yes, prompts for a commit message with that combined text as the default — press enter to accept it as-is, or type your own to override. The squash only ever touches commits not yet on `origin` (`git reset --soft` to the merge-base + one new commit), so the push right after is always a normal push, never a force-push — declining just pushes the commits individually as before.
   - **update-lock** (`u`, only offered when relevant) — shown when the repo itself owns a nested `gitman.yml` and at least one of *its own* declared deps has drifted: it has a recorded lock (`sources_locked`) but the actually-installed revision no longer matches it (someone committed/checked out something in that dep without re-locking). Confirms, then runs gitman's own `lock()` to write the currently-installed revisions back into `sources_locked` — a purely local operation, not a fetch/update of the deps themselves. A subproject with a stale lock but an otherwise clean, fully-pushed working tree still shows up in `review` specifically because of this. (A dep with no lock at all isn't considered stale — it just doesn't use gitman's locking.)
   - **skip** — leaves the repo untouched and moves to the next one.
   - **quit** — stops the review immediately.

Because revert always goes through a labeled stash instead of `git reset --hard`/`git clean`, and commit always shows the full file list (new files included) before asking for confirmation, no uncommitted work is ever silently discarded or swept into a commit by this command. Pushing always asks first too, since it's the one action here that touches something outside your machine.

Pass `-c`/`--use-cache` to skip the rescan (including the per-repo `fetch`) and reuse the snapshot written by the last `check`/`list` run — useful if you already saw the status and just want to act on it without waiting again. The list of *which* repos are dirty comes from that cached snapshot, but the diff shown and the commit/revert action itself always operate on the live working tree, so nothing stale is ever committed or stashed. If the cache is old, a repo that's changed since may not show up (or may show a diff different from what the cache implied) — rerun without `-c` if in doubt.

**`depman check -r`** runs the normal full status table first — so you see the whole project's state in one shot, the same overview `check -l -t` gives you — and then drops straight into the same review workflow, reusing that same scan instead of scanning twice.

**Processing order** (`--order`, available on both `review` and `check -r`):

- `root-last` (default) — nested dependency repos first (alphabetically), the root project repo last. Useful when the root's own state (e.g. `gitman.yml`, lockfiles) should be handled only after its dependencies have settled.
- `root-first` — alphabetical order, root repo (`.`) first.
- `gitman` — deps in the same depth-first, gitman.yml-declared order as the status table (see above), root last. Since you control that order just by reordering `sources` in `gitman.yml`, this is the one to use if you want "which repo do I commit/push first" to be something you can set explicitly rather than accepting alphabetical order — e.g. move a dep to the top of `gitman.yml` to make `review --order gitman` reach it first.

### `depman hooks` — installable pre-push warning hook

A normal `git push` (run directly, not through depman) can be made to warn about issues in a project's gitman-declared dependencies before it goes through:

- `depman hooks install [--force]` — writes `.git/hooks/pre-push` in the resolved `--root`. The script bakes in the absolute path of the Python interpreter running `depman hooks install` (so it works regardless of what's on `PATH`/which shell triggers the push — reinstall if you move environments) but deliberately does *not* bake in the repo path, so it stays valid if the repo itself is moved. Refuses to overwrite an existing hook unless `--force` is given, and never touches a hook it didn't install (checked via a marker comment).
- `depman hooks uninstall` — removes it, but only if depman installed it.
- `depman hooks check` — the actual check (also callable directly to preview what the hook would say without pushing anything). Walks the same gitman-declared dependency tree as everything else, but is deliberately **local-only — it never fetches** (unlike `check`/`list`/`review`), since git blocks the push while a pre-push hook runs and this needs to stay fast. It only reports uncommitted changes and unpushed-commit counts already knowable from what's on disk.
- **Always exits 0.** This warns, it never blocks a push — there's currently no option to make it fail the push instead; that'd be a natural, easy follow-up (e.g. a `--strict` flag) if wanted later.

## Automation (scripts / LLM agents)

`depman check --json` and `depman list --json` print a single JSON document to stdout instead of the human-oriented table/text output — nothing else is written to stdout in this mode, so it's safe to pipe (e.g. `depman check --json | jq .`). All diagnostic/progress output (the repo-scan progress bar, "✅ Dumped ..." cache messages, per-function timing) goes to stderr in every mode, not just `--json`, so redirecting stderr separately (`2>err.log`) always gets you clean data on stdout.

Shape: `{"configs": {...}, "repos": {...}}`.

- `configs.configs[project_path]` — one entry per loaded config (`project_path` is `.` for the root, or a relative path like `deps/widget` for nested configs): `location`, `config_file`, `rev_installed`, `backend` (which config backend produced this entry — see "Config backends" below), and `deps[dep_path]` with `name`, `repo`, `rev`, `rev_locked`, `rev_installed`, `in_repos`, `rev_match`, `behind_main` (the last three come from cross-referencing the dep against its actual git repo — see `analyze_configs_repos` in `depman/utils/configs.py`).
- `repos.repos[repo_path]` — one entry per discovered git repo (`repo_path` relative to `--root`, `.` for the root repo itself): `name`, `rev`, `rev_short`, `current_branch`, `remote_url`, `tags`, `commit_info`, `has_uncommitted`, `uncommitted_files`, `has_unpushed`, `unpushed_count`, `has_update`/`update_details`, `has_update_main`/`update_details_main`, `used_in_configs`, `in_config`.

This is read-only status reporting — there's currently no non-interactive equivalent of `depman review`'s commit/revert/push actions (that's fully interactive today); scripting those still means driving `review`'s prompts directly.

## Config backends

Discovering "where are the declared and locked dependency revisions for this project" is behind a small pluggable interface (`depman/backends/`, `ConfigBackend`: `discover()`, `parse_one()`, `relock()`), so depman isn't hard-wired to gitman specifically. **Only `gitman.yml` is implemented today** (`GitmanBackend`); everything downstream (git-repo scanning, cross-referencing, ordering, the status table, `review`, `hooks`) only depends on every config entry sharing the same shape and never needs to know which backend produced it. Backend detection is per config location, not per whole tree, so a root using one convention could in principle contain a subproject using another once a second backend exists. A git-submodules backend is a natural next implementation (GitPython's `Submodule` API already maps cleanly onto the same shape: `.hexsha` is the recorded/locked SHA, `.module().head.commit.hexsha` is what's actually installed) but isn't built yet.

## Limitations

- **Terminal mode is Windows-only** — uses `wt.exe`; no Linux/macOS equivalent implemented yet
- Caching writes `.cache_git_repos.yaml` and `.cache_configs.yaml` to the project root (ignored by git)
- Only the `gitman.yml` config backend is implemented; see "Config backends" above

## Performance

A live scan (`check`/`list`/`review` without `-c`) fetches every discovered repo from its remote — on large dependency trees this is the dominant cost, not CPU work. Repos are fetched/scanned concurrently (default 8 at a time; tune with `-j`/`--jobs`). If your git server rate-limits concurrent connections, lower `-j`; if you have many repos and headroom, raise it. Cross-referencing repos against configs (`analyze_configs_repos`) reuses the same fetch `find_all_git_repos` just did rather than fetching again, so it adds no extra network time. Use `-c`/`--use-cache` to skip the network scan entirely and reuse the last snapshot when you don't need fresh remote state.

## TODO / Future work

- [ ] Cross-platform terminal support (Linux: `gnome-terminal`/`konsole`, macOS: `Terminal.app`/`iTerm2`)
- [ ] Optional `--strict`/blocking mode for the pre-push hook (`depman hooks check` currently always exits 0)
- [ ] A git-submodules `ConfigBackend` implementation (see "Config backends" above)

## Recently fixed

- Update mode now searches all loaded configs (root and nested) for a dep's owning config, instead of only the root `.` config
- `analyze_configs_repos` is wired into the `check`/`list` flow; `rev_match`/`behind_main`/`in_repos` are computed for every config dep, and deps behind `origin/main` are reported in `list` output
- `pyperclip` and `pyyaml` added to `pyproject.toml` dependencies (previously missing, breaking a fresh install)
- Removed dead code: `find_all_git_repos1`, `scan_gitman_projects`, `depman/core/` (unused empty package)
- Real pytest suite covering `find_all_git_repos`, `find_all_configs`, and CLI `check`/`list` invocation
- `depman gm lock` and `depman gm uninstall` wrappers added
- `analyze_configs_repos` no longer re-fetches every dep's remote (it was redundantly refetching what `find_all_git_repos` had just fetched); `find_all_git_repos` now scans/fetches repos concurrently (`-j`/`--jobs`, default 8) instead of one at a time
- Status table / `list` row order is now deterministic (root first, then gitman.yml-declared depth-first order, undeclared repos last) instead of raw scan-completion order, which became nondeterministic once scanning went concurrent; list-mode's `select #:` index was fixed to match
- `depman review` no longer writes `.cache_*.yaml` files into the repo it's reviewing (it now reuses `check`/`list`'s scan machinery internally but opts out of the cache-write side effect, which was dirtying the very repo being reviewed with new untracked files)
- `review`'s commit step no longer crashes the whole session if a repo's `post-commit` hook fails (e.g. a repo configured for git-lfs where `git-lfs` isn't on `PATH`) — the commit itself already succeeded by the time such a hook runs (`post-commit` is documented as advisory-only), so this is now reported as a warning and the review continues
- `review` now suggests relocking (`u`) a subproject's `gitman.yml` when its lock has drifted from what's actually installed, and picks that up as a reason to include an otherwise-clean, fully-pushed subproject in the review at all
- `depman hooks install`/`uninstall`/`check` added — an optional, local-only, warn-only pre-push hook
- Config parsing/discovery (previously `gitman.yml`-parsing logic inlined in `find_all_configs`) is now behind a pluggable `ConfigBackend` interface (`depman/backends/`); pure refactor, verified behavior-identical against the full test suite — see "Config backends" above

## Development

```bash
pip install -e ".[dev]"
black . && flake8
pytest
```

MIT License. Contributions welcome.
