---
id: TASK-5
title: Migrate CLI framework from Click to Typer
status: Done
assignee:
  - ivansmel@gmail.com
created_date: '2026-09-24 13:46'
updated_date: '2026-09-24 14:36'
labels: []
dependencies: []
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
depman's CLI is currently built on Click across depman/cli.py and the command modules (checker.py, gm_commands.py, hooks.py, review.py) plus click usage in depman/utils/configs.py. Typer is built on Click but gives a more modern, type-hint-driven API (less boilerplate for options/arguments, built-in shell completion, cleaner subcommand composition) and is the framework the user wants depman to standardize on going forward.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 depman's CLI entry point and all subcommands (checker, gm_commands, hooks, review) are implemented with Typer instead of raw Click decorators
- [x] #2 All existing CLI options, arguments, help text, and subcommand behavior are preserved (no regressions in flags, defaults, or output)
- [x] #3 click.echo/click.style/click.confirm/click.prompt usages needed for interactive review UX are still supported (via Typer/Click's underlying click.Context and rich/click echo helpers) with unchanged interactive behavior
- [x] #4 Existing CLI tests in tests/test_depman.py pass against the Typer-based CLI
- [x] #5 click is removed from direct usage in favor of typer where Typer provides an equivalent, or documented as an intentional remaining dependency where it doesn't
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add typer to pyproject.toml dependencies (>=0.12, matching the installed 0.17.3 in the env_dep env); keep click as a dependency since Typer is built on it and click.Choice/click.ClickException remain intentionally in use (see step 6).
2. Rewrite depman/commands/gm_commands.py: replace the click Group/commands with a gm_app = typer.Typer(name='gm', ...); convert each @gm.command to a plain function decorated with @gm_app.command(name), using typer.Context (auto-injected, no @click.pass_context needed) and typer.Argument/typer.Option for names/depth/force/recurse/lock/keep-location, preserving every flag name, short alias, and default exactly.
3. Rewrite depman/commands/hooks.py similarly: hooks_app = typer.Typer(name='hooks', ...); hooks_install/hooks_uninstall/hooks_check become @hooks_app.command()-decorated functions with typer.Context/typer.Option. Internal click.echo/click.style calls become typer.echo/typer.style (same functions, re-exported by typer) -- pure rename, no behavior change.
4. Rewrite depman/commands/checker.py: check_cmd and list_cmd become plain functions (not decorated in-module, matching the existing cli.add_command() attach pattern) taking typer.Context plus typer.Option-annotated params for every existing flag (--use-cache/-c, --update-mode/-u, --list-mode/-l, --terminal-mode/-t, --review/-r, --order via click_type=click.Choice(ORDER_CHOICES) since Typer has no native equivalent for an ad-hoc string-list choice outside of Enums, --json/as_json, --jobs/-j). click.echo -> typer.echo.
5. Rewrite depman/commands/review.py: keep all helper functions (_commit_repo, _review_repo, etc.) but swap click.echo/click.style/click.confirm/click.prompt for their typer equivalents (typer.echo/typer.style/typer.confirm/typer.prompt -- same underlying functions); click.prompt's type=click.Choice(choices) for the interactive c/r/d/p/u/s/q menu stays as click.Choice (documented intentional click usage -- Typer has no interactive-prompt Choice type of its own). review_cmd becomes a plain function with typer.Context/typer.Option params for --use-cache/-c, --order, --jobs/-j, still calling the unchanged run_review().
6. Rewrite depman/cli.py: build a typer.Typer app with context_settings enabling -h as a help alias; move the root --root option and its find_git_root-or-raise logic into an @app.callback() function setting ctx.obj (root-not-found case raises click.ClickException, kept as a documented intentional click usage since Typer has no equivalent that reproduces click's Error-message-plus-exit-1 formatting); attach gm_app and hooks_app via app.add_typer, and check_cmd/list_cmd/review_cmd via app.command(name)(func); expose cli = typer.main.get_command(app) so the pyproject.toml script entry point and tests/test_depman.py's existing CliRunner-based invocation of cli keep working completely unchanged -- get_command() converts the Typer app into an ordinary click.Group.
7. Convert depman/utils/configs.py's direct click.echo/click.style/click.getchar usages (pure output/prompt helpers, no command decorators there) to typer.echo/typer.style/typer.getchar for consistency with AC#5's intent to remove click in favor of typer where it has an equivalent; leave import click only if something without a typer equivalent remains there (verify during implementation).
8. Run the full existing test suite (tests/test_depman.py) unchanged first to catch behavior regressions; fix any option-parsing mismatches (e.g. flag defaults, nargs=-1 -> List[str] Argument behavior for gm install/update names) surfaced by failures.
9. Manually smoke-test --help, -h, gm --help, hooks --help, review --help, and check --json via CliRunner to confirm the CLI still resolves and prints reasonable help/output (Typer's rich-formatted --help output differs cosmetically from click's plain text, which is expected and not a regression per AC#2's intent -- flags/defaults/behavior, not exact help formatting).
10. Update README if any documented command invocation syntax changed (it shouldn't -- CLI surface stays identical), and add a Recently-fixed changelog entry noting the Click-to-Typer migration plus the remaining intentional click dependency (click.Choice / click.ClickException) for AC#5.
11. Run the full test suite one final time to confirm everything passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented: added typer>=0.12.0 to pyproject.toml (kept click as a dependency, still needed). Rewrote depman/commands/gm_commands.py, hooks.py, checker.py, review.py to define commands as plain functions using typer.Context/typer.Option/typer.Argument instead of @click.command/@click.option/@click.pass_context; bulk-renamed click.echo/style/confirm/prompt to typer.echo/style/confirm/prompt throughout review.py and depman/utils/configs.py (same underlying functions, Typer re-exports them). depman/cli.py now builds a typer.Typer() app, attaches gm_app/hooks_app via app.add_typer and check_cmd/list_cmd/review_cmd via app.command(name)(func), then exposes cli = typer.main.get_command(app) -- converts the Typer app back into an ordinary click.Group so the existing pyproject.toml script entry point and tests/test_depman.py's click.testing.CliRunner-based invocation needed zero changes. Remaining intentional click usage (documented in README): click.Choice (for --order and the interactive review menu's click.prompt type=) and click.ClickException (for the no-git-root error) -- Typer has no native equivalent for either. Verified: full suite 61 passed unchanged; manual CliRunner smoke test of --help/-h on the root app and every subcommand (gm init/install/update/list/lock/uninstall, hooks install/uninstall/check, check, list, review) confirmed all flags/args/short-aliases survived and -h works everywhere (including gm/hooks subcommands, which never had -h before -- a bonus from Typer's context_settings inheritance down the whole command tree, not a regression). Updated README with a Recently-fixed entry describing the migration and the two documented remaining click usages.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated depman's entire CLI from raw Click decorators to Typer. depman/commands/gm_commands.py, hooks.py, checker.py, and review.py now define commands as plain functions using typer.Context/typer.Option/typer.Argument; depman/cli.py builds a typer.Typer() app (app.add_typer for the gm/hooks sub-apps, app.command(name)(func) for check/list/review) and exposes cli = typer.main.get_command(app), converting it back into a plain click.Group so the pyproject.toml 'depman = depman.cli:cli' entry point and tests/test_depman.py's existing click.testing.CliRunner invocation needed no changes at all. click.echo/style/confirm/prompt calls in review.py and depman/utils/configs.py were renamed to their typer equivalents (same underlying functions). Two intentional click usages remain and are documented in README's changelog: click.Choice (--order option and the interactive review menu's prompt type) and click.ClickException (the no-git-root error) -- Typer has no native equivalent for either. Verified with: (1) grep confirming zero @click.* decorators remain anywhere in depman/; (2) the full test suite, 61/61 passing unchanged; (3) a manual CliRunner smoke test of --help/-h against the root app and every subcommand (gm init/install/update/list/lock/uninstall, hooks install/uninstall/check, check, list, review), confirming every flag, argument, and short alias survived, with -h now also working on gm/hooks subcommands as a bonus (Typer's context_settings inherits down the whole command tree, unlike the original code which never wired that up for those two groups).
<!-- SECTION:FINAL_SUMMARY:END -->
