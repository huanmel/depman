---
id: TASK-6
title: Add `depman release` command
status: Done
assignee:
  - '@claude'
created_date: '2026-10-09 08:38'
updated_date: '2026-10-09 08:41'
labels: []
dependencies: []
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cutting a release is a manual sequence (lock, bump version file, run custom scripts, commit, tag, push with tags). Automating it with confirmations removes mistakes, and sometimes an existing tag must be force-moved to a new commit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 release accepts major|minor|fix(patch) or an explicit version; bump is resolved from the latest semver tag reachable from HEAD (v0.0.0 if none)
- [x] #2 Resolved version, scripts, files and push command are printed and confirmed before lock/scripts, before commit, and before push
- [x] #3 Runs gitman lock, version scripts, pre-commit scripts (with {version}/{tag} substitution and DEPMAN_VERSION/DEPMAN_TAG env), commits all changes, creates an annotated tag, pushes branch with --follow-tags
- [x] #4 --force-tag moves an existing tag locally and force-pushes only that tag to the remote, after explicit confirmation
- [x] #5 Scripts configurable via --version-script/--pre-commit-script flags and a release section in .depman.yml; tests cover version resolution and the end-to-end flow
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added depman release (depman/commands/release.py): version resolution from reachable semver tags, three confirmations, lock/version/pre-commit scripts, annotated tag, push --follow-tags, --force-tag, --dry-run, .depman.yml config. Verified with tests/test_release.py (full suite: 71 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
