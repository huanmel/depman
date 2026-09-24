---
id: TASK-2
title: Prompt to gitman-lock unlocked deps before committing a subproject
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24 14:15'
updated_date: '2026-08-24 14:25'
labels: []
dependencies: []
references:
  - depman/commands/review.py
  - depman/backends/gitman_backend.py
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
review's commit step (_commit_repo in depman/commands/review.py) already offers to relock a subproject's gitman.yml when a dep IS locked but drifted (_stale_locked_deps / the 'u' action, rev_locked is not None but mismatched). Deps that were NEVER locked at all (rev_locked is None) are deliberately excluded from that check today -- they're treated as 'this project doesn't use locking' rather than stale. But when a subproject's own gitman.yml declares deps with no lock at all, that's often just an oversight (nobody ever ran gitman lock for it), and the moment you're about to commit in that subproject is a good moment to catch it. Add a check, specific to the commit step, that looks at the repo's own config for deps with rev_locked is None and offers to run gitman lock (via the same ConfigBackend.relock() used by the existing stale-lock 'u' action) before the commit proceeds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When about to commit a repo whose own gitman.yml has one or more deps with no lock at all (rev_locked is None), review's commit step asks whether to run gitman's lock() for those deps before proceeding with the commit
- [x] #2 Confirming locks those deps via the same backend relock() mechanism as the existing stale-lock update action, and the resulting gitman.yml change is included in the same commit that follows
- [x] #3 Declining does not block the commit -- it proceeds normally without locking
- [x] #4 A repo with no gitman.yml of its own, or whose gitman.yml has no unlocked deps (all already locked, or no deps declared), does not show this prompt
- [x] #5 Covered by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add _relock_config(repo, repo_path, dep_names, configs) -> bool in review.py: extracts the backend-lookup + backend.relock() call + success/failure messaging + in-memory config refresh (rev_match=True, rev_locked=rev_installed for every dep in that config, since gitman.lock() relocks the whole file regardless of dep_names -- confirmed by reading GitmanBackend.relock) that's currently duplicated inline in _update_subproject_lock.
2. Refactor _update_subproject_lock to call _relock_config after its existing confirm (pure refactor, no behavior change for the existing stale-lock 'u' action).
3. Add _unlocked_deps(configs, repo_path) -> List[str]: mirrors _stale_locked_deps but selects deps where rev_locked is None -- the never-locked case _stale_locked_deps deliberately excludes.
4. Add _offer_lock_unlocked_deps(repo, repo_path, configs): if _unlocked_deps is non-empty, prints them and confirms (default False) 'Run gitman lock for {repo_path} before committing?'; on yes calls _relock_config.
5. Call _offer_lock_unlocked_deps at the top of _commit_repo, then always recompute files = _changed_files(repo) afterward so a fresh lock is reflected in the file list/confirm/commit that follows, before the existing new_files/print/confirm logic.
6. Add tests: a fixture with a subproject whose gitman.yml has a genuinely unlocked dep (no sources_locked at all); a test that committing prompts to lock and confirming includes the lock in the same commit; a test that declining still commits without locking; reuse existing fixtures to prove no prompt appears for a repo with no config or no unlocked deps.
7. Update README's review section and changelog.
8. Run the full test suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted _relock_config from _update_subproject_lock (pure refactor); added _unlocked_deps and _offer_lock_unlocked_deps; wired the offer into the top of _commit_repo with a fresh _changed_files() recompute afterward so a resulting lock rides along in the same commit. Discovered the 'project' fixture's root gitman.yml declares 'widget' unlocked, so 5 pre-existing review-commit tests needed an extra decline input line for the new prompt -- fixed, all still pass with unchanged assertions. Added unlocked_leaf_project fixture plus 3 tests: lock offer accepted (lock rides into the commit), declined (commit still proceeds, dep stays unlocked), and no offer when the dep is already locked (reusing locked_leaf_project). Full suite: 58 passed (55 prior + 3 new), env_dep interpreter. README's review section and changelog updated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
review's commit step now checks whether the repo being committed owns a gitman.yml with any never-locked dep (rev_locked is None) and, before proceeding, offers to run gitman's lock() for it -- accepting folds the resulting sources_locked entry into the same commit, declining never blocks the commit. Implemented via _unlocked_deps/_offer_lock_unlocked_deps in depman/commands/review.py, sharing a new _relock_config helper with the existing stale-lock 'u' action (pure refactor of _update_subproject_lock, no behavior change there). Verified with 3 new tests plus the full 58-test suite (all passing) using real git.Repo fixtures through CliRunner; also caught and fixed 5 pre-existing tests whose fixture repo had an unlocked dep and needed an extra input line for the new prompt.
<!-- SECTION:FINAL_SUMMARY:END -->
