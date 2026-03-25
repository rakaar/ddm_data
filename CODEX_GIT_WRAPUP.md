# Codex Git Wrapup Shortcut

Use this file as a shorthand reference when you want the standard repository wrapup.

If you tell Codex something like:

- "Do [CODEX_GIT_WRAPUP.md](/home/rlab/raghavendra/ddm_data/CODEX_GIT_WRAPUP.md)"
- "Please follow [CODEX_GIT_WRAPUP.md](/home/rlab/raghavendra/ddm_data/CODEX_GIT_WRAPUP.md)"

interpret that as:

1. Run `git status` and inspect the current worktree.
2. Find any new or undocumented files touched by the current work.
3. Document those files in the appropriate local `README.md` files.
4. Commit all intended changes with a sensible commit message.
5. Push the commit to `origin/main`.

Default expectations:

- Do not revert unrelated user changes.
- Keep existing README entries accurate if a file's behavior changed materially.
- Mention which files were documented, the commit hash, and whether push succeeded.
