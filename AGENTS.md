# Repository Preferences

- In `fitting_aborts/`, prefer script structure with `# %%` cell blocks for interactive execution.
- Avoid `if __name__ == "__main__":` / `main()` wrappers unless explicitly requested.
- Keep parameters near the top in a dedicated `# %%` section so they are easy to edit per run.
- When running Python locally, use `.venv/bin/python`.
- Always invoke scripts with the virtual environment interpreter (for example: `.venv/bin/python fitting_aborts/aborts_animal_wise_explore.py`).
