---
id: TASK-3
title: Unify the pre-commit lock gate to also cover stale (not just unlocked) deps
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24 16:54'
updated_date: '2026-08-24 17:03'
labels: []
dependencies: []
references:
  - depman/commands/review.py
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2 added a pre-commit gate for deps that have never been locked at all (rev_locked is None): _offer_lock_unlocked_deps, called at the top of _commit_repo in depman/commands/review.py. Deps that ARE locked but stale (rev_locked set, mismatched against what's installed -- _stale_locked_deps) are only surfaced separately: an informational warning line plus the top-level 'u' menu choice in _review_repo. Typing 'c' straight away, without ever picking 'u' first, commits without ever offering to relock a stale dep. Confirmed with the user (2026-08-24) that this asymmetry should be closed: both unlocked and stale deps in a repo's own gitman.yml should trigger the same pre-commit prompt, since gitman's lock() relocks the whole file in one call regardless of which subset needed it. The existing top-level 'u' choice stays as-is for relocking a stale dep without committing anything else (e.g. an otherwise clean, fully-pushed subproject).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Committing a repo whose own gitman.yml has a stale-locked dep (rev_locked set but mismatched) offers to run gitman lock before the commit proceeds, the same way an unlocked dep does today
- [x] #2 A repo with both unlocked and stale deps shows one combined prompt/relock, not two separate ones
- [x] #3 Declining still does not block the commit
- [x] #4 The existing top-level 'u' menu action for stale-only repos (clean working tree, nothing else to commit) is unchanged
- [x] #5 Covered by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace _offer_lock_unlocked_deps with _offer_lock_before_commit(repo, repo_path, configs) in review.py: combines _unlocked_deps(configs, repo_path) and _stale_locked_deps(configs, repo_path) into one list; if either is non-empty, prints a combined message (unlocked count/names, stale count/names) and asks one 'Run gitman lock ... before committing?' confirm; on yes calls the existing _relock_config once (gitman.lock() relocks the whole file regardless of which subset needed it, already confirmed when _relock_config was written for TASK-2).
2. Update _commit_repo's call site to use the renamed function; no other changes needed there (the existing files = _changed_files(repo) refresh after the offer already covers both cases).
3. Leave _review_repo's existing stale_deps warning line + top-level 'u' menu choice untouched -- that path stays for relocking a stale-only repo with nothing else to commit.
4. Add tests: a repo with only a stale dep triggers the new pre-commit prompt on 'c' (not just via 'u'); a repo with both an unlocked dep and a stale dep shows one combined prompt/relock, not two; the existing stale-only subproject_project-based 'u' tests continue to pass unchanged, proving the top-level action still works standalone.
5. Update README's review section and changelog to describe the unified check.
6. Run the full test suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed _offer_lock_unlocked_deps to _offer_lock_before_commit and broadened it to combine _unlocked_deps + _stale_locked_deps into one message/confirm/relock call. Existing top-level 'u' action and its warning line in _review_repo untouched -- verified unchanged by the pre-existing subproject_project-based relock tests still passing with the same input sequences. Discovered the message text changed ('unlocked dep(s): X' -> 'N unlocked (X)' / 'N stale (Y)' combined phrasing), so updated 2 existing TASK-2 test assertions to match. Added test_review_commit_offers_to_lock_stale_deps_too (subproject_project, stale-only, commits via plain c) and test_review_commit_combines_unlocked_and_stale_into_one_prompt (new mixed_lock_project fixture: one unlocked + one stale dep in the same gitman.yml, asserts exactly one 'Run gitman lock' prompt). Full suite: 60 passed (58 prior + 2 new), env_dep interpreter. README's review section and changelog updated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the asymmetry where review's pre-commit lock offer (added by TASK-2) only covered never-locked deps: committing a repo with a stale-but-locked dep now also triggers the same offer, in one combined prompt when both kinds are present. Implemented by broadening _offer_lock_unlocked_deps into _offer_lock_before_commit(repo, repo_path, configs) in depman/commands/review.py, which unions _unlocked_deps and _stale_locked_deps before offering the single relock via _relock_config. The standalone top-level 'u' menu action (for relocking a stale-only, otherwise clean/pushed repo with nothing to commit) is untouched. Verified with 2 new tests plus the full 60-test suite (all passing) using real git.Repo fixtures through CliRunner; also fixed 2 pre-existing TASK-2 tests whose assertions depended on now-superseded message wording.
<!-- SECTION:FINAL_SUMMARY:END -->
