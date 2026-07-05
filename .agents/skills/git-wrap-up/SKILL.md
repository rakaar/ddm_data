---
name: git-wrap-up
description: Standard repository wrap-up workflow for Codex. Use when the user says "git wrap up", asks to follow CODEX_GIT_WRAPUP.md, or otherwise asks Codex to inspect the current git worktree, document new or changed files, commit intended changes, and push to origin/main.
---

# Git Wrap Up

## Workflow

Treat "git wrap up" as the standard repository close-out:

1. Run `git status` and inspect the current worktree.
2. Identify which changed, new, or undocumented files belong to the current task.
3. Leave unrelated user changes untouched and unstaged unless the user explicitly includes them.
4. Document new or materially changed files in the appropriate local `README.md` files.
5. Run focused validation for the intended changes when practical.
6. Stage only the intended files.
7. Commit with a concise, sensible message.
8. Push the commit to `origin/main`.

## Reporting

In the final response, mention:

- Which files were documented.
- Which unrelated changes were left alone.
- The commit hash.
- Whether the push succeeded.
- Any validation that was run or could not be run.
