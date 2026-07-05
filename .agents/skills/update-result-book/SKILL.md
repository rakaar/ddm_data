---
name: update-result-book
description: Update this repository's MkDocs result book when the user asks to add, log, document, or publish a generated result figure. Use for workflows where an agent ran an analysis script or notebook, produced a figure, and needs to add that figure with a caption and the source Python/notebook path to the date-wise result docs.
---

# Update Result Book

## Overview

Use this skill to add generated research figures to the repository's MkDocs result book. The book is organized by date, with one Markdown page per date.

## Repository Layout

- MkDocs config: `mkdocs.yml`
- Daily result pages: `docs/results/YYYY-MM-DD.md`
- Result assets: `docs/assets/results/YYYY-MM-DD/`
- Result index: `docs/results/index.md`

## Workflow

1. Identify the generated figure path and the Python file or notebook that generated it.
2. Use the local date unless the user asks for a specific date.
3. Put copied figure assets in `docs/assets/results/YYYY-MM-DD/`.
4. Add or update `docs/results/YYYY-MM-DD.md`.
5. Ensure `docs/results/index.md` and `mkdocs.yml` include the date page.
6. Build the docs with `.venv/bin/python -m mkdocs build --strict` from the repository root.
7. Do not run an immediate Google Drive backup or update `FIT_BACKUP_LEDGER.md` unless the user explicitly asks for a Drive backup. The repository's scheduled cron backup is the default persistence path.

Each result entry should include the figure, a concise caption, and the exact source path:

```markdown
## Short result title

![Short alt text](../assets/results/YYYY-MM-DD/figure_name.png)

*Caption describing the result and the important condition/model/data shown.*

Source: `relative/path/to/script.py`
Figure: `docs/assets/results/YYYY-MM-DD/figure_name.png`
```

## Helper Script

For routine additions, prefer the bundled script:

```bash
.venv/bin/python .agents/skills/update-result-book/scripts/update_result_book.py \
  --repo /path/to/ddm_data \
  --figure path/to/generated.png \
  --source path/to/script.py \
  --title "Short result title" \
  --caption "Caption for the figure."
```

The script copies the figure into the date-specific assets folder, appends the Markdown entry, and updates the result index and MkDocs navigation. After using it, inspect the diff and run the MkDocs build command.

## Google Drive Backup

Immediate Google Drive backup is optional and should only be done when the user explicitly asks for it, for example "backup the result book to Drive now." Otherwise, rely on the repository's scheduled cron backup.

When an explicit immediate backup is requested:

1. Check for `raga:` with `rclone listremotes`.
2. Use `rclone copy`, not `rclone sync`.
3. Back up at least:
   - `docs/`
   - `mkdocs.yml`
   - `RESULT_BOOK_AGENT_SETUP.md` if present
   - `FIT_BACKUP_LEDGER.md` if present
   - `scripts/backup_result_book_to_drive` if present
   - this skill folder if it was changed
4. Use a dated Drive folder such as `raga:ddm_result_book_backups_YYYYMMDD_lavos/`.
5. Verify with `rclone lsf -R --files-only DRIVE_PATH | wc -l` and `rclone size DRIVE_PATH`.
6. Append a short entry to `FIT_BACKUP_LEDGER.md` when present.
