# Local Coding Rules

## Preferred code style for analysis scripts
- Prefer notebook-cell style Python scripts using `# %%` sections so code can be run and tested cell-by-cell.
- Avoid `if __name__ == "__main__":` wrappers for exploratory/plotting workflows unless explicitly requested.
- Keep data loading, processing, and plotting in separate runnable cells.
