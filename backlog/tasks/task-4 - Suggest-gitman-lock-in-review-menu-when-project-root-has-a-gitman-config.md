---
id: TASK-4
title: Suggest gitman lock in review menu when project root has a gitman config
status: Done
assignee:
  - ivansmel@gmail.com
created_date: '2026-09-24 13:46'
updated_date: '2026-09-24 14:09'
labels: []
dependencies: []
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The review command already offers a gitman-lock prompt when its dep-parsing (get_configs_and_repos) detects unlocked or stale deps for a repo (see _offer_lock_before_commit / stale_deps handling in depman/commands/review.py). That path depends on the gitman.yml having been successfully parsed into the configs structure. The user wants a simpler, direct trigger: when the root of the project currently being reviewed has its own gitman config file present, the review menu should suggest running 'gitman lock' there, independent of whatever the dep-parsing path already surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reviewing a project whose root contains a gitman config file surfaces a suggestion/option in the review menu to run 'gitman lock'
- [x] #2 No such suggestion appears when the project root has no gitman config file
- [x] #3 Tests cover both the presence and absence of a root gitman config file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In depman/commands/review.py add _has_config(configs, repo_path) -> bool: True iff repo_path has an entry in configs['configs'] (a config file -- currently only gitman.yml -- was discovered there), independent of dep count or lock state.
2. Extract _describe_pending_lock(configs, repo_path) -> Optional[str] from _offer_lock_before_commit's parts-building logic (the 'N unlocked (...); N stale (...)' summary text). Refactor _offer_lock_before_commit to use it -- no behavior/text change for the existing pre-commit gate.
3. Rename _update_subproject_lock(repo, repo_path, stale_deps, configs) to _offer_manual_relock(repo, repo_path, configs): drop the stale_deps param, compute the summary via _describe_pending_lock. When a summary exists, keep the current 'Relock gitman.yml in {repo_path} for {summary}?' phrasing. When it's None (config file present but nothing unlocked/stale), show a distinct message ('no unlocked or stale deps right now -- run gitman lock anyway?'), default False either way; on confirm call _relock_config with the current unlocked+stale dep name list (possibly empty).
4. In _review_repo, gate choices.append('u') / the '/ update-lock' prompt label on _has_config(configs, repo_path) instead of stale_deps, so the option appears whenever the repo owns a config file regardless of lock state. Leave default_choice's 'u' branch tied to stale_deps only (unchanged) so a clean, fully-synced repo doesn't silently default to relocking -- the option becomes visible/selectable but isn't forced. Update the 'u' branch call site to _offer_manual_relock(repo, repo_path, configs).
5. Update tests/test_depman.py: adjust test_review_no_stale_lock_warning_when_never_locked (drop its now-inverted 'update-lock not in output' assertion since root's own gitman.yml means the option is now expected); add a test using locked_leaf_project proving 'update-lock' appears and works (via the new no-pending message) even though sub's lock is fully in sync; add a test using a standalone repo with no gitman.yml at all proving 'update-lock' never appears.
6. Update README's review section (update-lock bullet) to describe the broadened, file-presence-based trigger, and add a 'Recently fixed' entry.
7. Run the full test suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented: added _has_config() and _describe_pending_lock() helpers in review.py; refactored _offer_lock_before_commit to use the shared summary helper (no behavior change); renamed _update_subproject_lock to _offer_manual_relock, which now handles the 'nothing pending' case with its own confirm message; _review_repo's top-level 'u'/update-lock menu option now gates on _has_config(configs, repo_path) instead of stale_deps (default_choice still tied to stale_deps only, so a synced repo doesn't silently default to relocking). Updated test_review_no_stale_lock_warning_when_never_locked (dropped its now-inverted update-lock assertion) and added test_review_update_lock_option_availability covering both presence (locked_leaf_project's fully-synced 'sub') and absence (a standalone repo with no gitman.yml) cases. Updated README's update-lock bullet and added a Recently fixed entry. Full suite: 61 passed (60 prior + 1 new), env_dep interpreter.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
review's top-level 'update-lock' (u) menu option now appears whenever a repo owns a gitman.yml at all -- root included -- not just when it already has a stale dep. Implemented via new _has_config()/_describe_pending_lock() helpers in depman/commands/review.py; _review_repo gates the 'u' choice/prompt-label on _has_config(configs, repo_path) (default_choice still keyed off stale_deps only, so a synced repo doesn't silently default to relocking); renamed _update_subproject_lock to _offer_manual_relock, which shows a distinct 'no unlocked or stale deps right now -- run anyway?' message when nothing is pending. Verified: (1) full test suite -- 61 passed (60 prior + 1 new: test_review_update_lock_option_availability, covering both a fully-synced config (locked_leaf_project's sub) showing 'update-lock' and a standalone repo with no gitman.yml never showing it); (2) a manual CliRunner probe against a project whose ROOT gitman.yml is dirty confirmed the literal root-of-project case: the printed menu is 'commit / revert / diff / push / update-lock / skip / quit'. Updated README's update-lock bullet and added a Recently fixed entry.
<!-- SECTION:FINAL_SUMMARY:END -->
