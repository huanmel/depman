---
id: TASK-1
title: Warn when committing a dependency invalidates a consuming config's lock
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 15:51'
updated_date: '2026-08-20 16:09'
labels: []
dependencies: []
references:
  - depman/commands/review.py
  - depman/backends/gitman_backend.py
type: enhancement
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Today, review's stale-lock detection (_stale_locked_deps in depman/commands/review.py) only fires when reviewing the PARENT/consuming config itself, comparing its rev_locked against the dependency's currently-installed rev. That only surfaces the problem after the fact, whenever someone happens to review or scan the consuming config later.

The scan already tracks, per dependency repo, which configs reference it and what rev they currently consider locked/installed (repos[dep_path]['used_in_configs'], populated by each config backend's parse_one -- see depman/backends/gitman_backend.py). review's commit action (_commit_repo in depman/commands/review.py) could use this proactively: when committing a repo that is itself a declared dependency in one or more other configs, check whether the new commit will leave that repo's rev different from what those consuming config(s) currently have locked, and warn the user right there -- catching the problem at the point it's created (committing in the dependency) instead of only when someone later reviews the parent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Committing a repo that is a declared dependency in one or more other configs, where the commit leaves the repo's rev different from what those config(s) currently have locked, displays a message listing which consuming config(s) will need relocking
- [x] #2 The warning does not block the commit
- [x] #3 A repo not referenced as a dependency by any other config produces no such warning
- [x] #4 A consuming config with no lock at all for that dependency (rev_locked is None -- unlocked, not stale) produces no such warning, consistent with existing _stale_locked_deps semantics
- [x] #5 Covered by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add _consuming_configs_needing_relock(configs, repo_path, new_rev) to review.py: scans every OTHER config's deps for repo_path, returns consuming config paths where rev_locked is set (not None) and != new_rev (mirrors _stale_locked_deps's null-check semantics but in the opposite direction -- repo_path as someone else's dependency, not its own children).
2. Add _warn_consuming_configs_stale(repo, repo_path, configs): reads repo.head.commit.hexsha as new_rev after a successful commit, calls the helper above, prints a non-blocking warning listing affected consuming configs if any.
3. Thread configs into _commit_repo(repo, repo_path, files, configs); call the warning helper after both the normal-success return and the HookExecutionError-recovered-success path (both are successful commits).
4. Update _review_repo's call site to pass configs into _commit_repo.
5. Add a fixture with a genuinely in-sync lock (sub locks leaf to leaf's actual current rev), then a test that commits a new change directly in leaf and asserts the warning fires naming the consuming config; a test using the existing unlocked project fixture asserting no warning; a test with a repo referenced by no config at all asserting no warning.
6. Update README's review section to document this.
7. Run the full test suite and do a manual end-to-end check against real repos.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented _consuming_configs_needing_relock and _warn_consuming_configs_stale in review.py; threaded configs into _commit_repo (both the normal-success and HookExecutionError-recovered-success paths). Added locked_leaf_project fixture (sub locks leaf to leaf's real current rev) plus 3 new tests: warning fires and names the consuming config, no warning when the consuming config is unlocked, no warning when the repo isn't referenced by any config. Full suite: 55 passed (52 pre-existing + 3 new), env_dep interpreter. README's review section and changelog updated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a proactive warning to review's commit step: committing a repo that is a locked dependency of one or more OTHER configs now checks, right after the commit, whether the new revision leaves those configs' locks stale, and if so names them. Implemented via _consuming_configs_needing_relock/_warn_consuming_configs_stale in depman/commands/review.py, using the existing configs snapshot (mirrors _stale_locked_deps but looks at consumers instead of a config's own children). Never blocks the commit. Verified with 3 new tests plus the full 55-test suite (all passing) using real git.Repo fixtures through CliRunner, not mocks.
<!-- SECTION:FINAL_SUMMARY:END -->
