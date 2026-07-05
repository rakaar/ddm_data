---
name: google-drive-fit-backup
description: Use when backing up DDM fit results or other large research fit outputs to Google Drive with rclone. Covers checking the rclone remote, creating Drive folders, running uploads in tmux with logs, verifying counts/sizes, and updating the fit backup ledger.
---

# Google Drive Fit Backup

Use this skill when the user asks to back up fit result folders to Google Drive, especially large DDM/VBMC outputs.

## Workflow

1. Confirm the rclone remote:

```bash
rclone listremotes
```

Prefer the existing `raga:` remote when present.

2. Inspect local source counts before upload:

```bash
find SOURCE_DIR -type f | wc -l
du -sh SOURCE_DIR
```

For structured fit folders, count important subfolders separately, such as upstream results, condition pickles, and corner plots.

3. Create a dated Google Drive parent folder and descriptive subfolders. Include the machine name in the folder names when the backup is machine-specific, for example:

```text
raga:ddm_fit_backups_YYYYMMDD_ganon/
```

4. Run long transfers inside tmux with a timestamped log:

```bash
tmux new-session -d -s SESSION_NAME 'cd /path/to/repo && ./scripts/backup_script.sh 2>&1 | tee -a backup_YYYYMMDD_HHMMSS.log'
```

Use `rclone copy`, not `sync`, unless the user explicitly asks for deletion mirroring.

Recommended options:

```bash
rclone copy SOURCE DEST --transfers 4 --checkers 8 --drive-chunk-size 64M --progress --stats 30s --log-level INFO
```

5. Verify remote counts after upload:

```bash
rclone lsf DRIVE_PATH --files-only | wc -l
rclone size DRIVE_PARENT
```

6. Update the repo ledger if present. Prefer `FIT_BACKUP_LEDGER.md` for this project. Include:

- what ran on which machine
- remote machine path
- local copied path
- Google Drive folder path
- expected and observed file counts
- upload status and log path

## Safety Notes

- Do not use `rclone sync` for backups unless explicitly requested.
- Keep uploads resumable/idempotent by using `rclone copy`.
- Put large transfers in tmux so they survive laptop disconnects.
- Do not expose or modify OAuth credentials; use the configured rclone remote.
