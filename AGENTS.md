# Repository Preferences

- In `fitting_aborts/`, prefer script structure with `# %%` cell blocks for interactive execution.
- Avoid `if __name__ == "__main__":` / `main()` wrappers unless explicitly requested.
- Keep parameters near the top in a dedicated `# %%` section so they are easy to edit per run.
- When running Python locally, use `.venv/bin/python`.
- Always invoke scripts with the virtual environment interpreter (for example: `.venv/bin/python fitting_aborts/aborts_animal_wise_explore.py`).


- This is exploratory research code, not production software.
- Prefer a small number of longer, readable blocks over many tiny helper functions in style of #%%.
Do not over-abstract.
- Avoid creating single-use functions just to make the code look cleaner; keep logic inline when it is only used once and is still easy to follow.
- Write a function when the same logic is used in multiple places, when a block is getting hard to read, or when you want to isolate a stable step that you may reuse or test later.
- Avoid thin wrapper functions. If a helper only forwards arguments to another helper, renames a value, or applies a trivial unit conversion, collapse it into one function unless both forms are genuinely reused.
- For unit conversions, prefer doing the conversion at the use site unless both unit systems are used in multiple places.
- If choosing between two small helpers and one slightly longer helper, prefer the single helper.
- Use only necessary checks: include checks for realistic failure modes in research workflows, but do not add defensive branches for unlikely edge cases that do not matter here.
- Optimize for easy modification and visibility of logic in one place.
