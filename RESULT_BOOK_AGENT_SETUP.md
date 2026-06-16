# Agent Prompt: Set Up MkDocs Result Book and Update Skill

Give this prompt to an agent in a research/code repository where you want a date-wise result book.

---

I want you to set up a MkDocs result book for this repository and create a Codex skill that future agents will use when I say something like "update the result docs", "add this figure to the result book", or "log this result".

## Goal

Set up a date-wise MkDocs site where each result page corresponds to one date. When an agent generates a figure from a Python script or notebook, the agent should be able to add that figure to the correct date page with:

- the figure embedded
- a short caption
- the path of the Python file or notebook that generated it
- the stored figure path inside the docs assets folder

## MkDocs Setup

Create this structure:

```text
mkdocs.yml
docs/
  index.md
  results/
    index.md
    YYYY-MM-DD.md
  assets/
    results/
      YYYY-MM-DD/
```

Use one Markdown page per date:

```text
docs/results/2026-06-16.md
docs/results/2026-06-17.md
...
```

Store result figures here:

```text
docs/assets/results/YYYY-MM-DD/
```

Add `mkdocs` to the project dependencies. If the repo has a virtual environment, use that environment's Python to install and run MkDocs.

Use a basic MkDocs theme unless the repo already has a preferred theme.

Set the MkDocs dev server to bind only on localhost:

```yaml
dev_addr: 127.0.0.1:8000
```

Quote date labels in `mkdocs.yml` navigation so YAML does not parse them as dates:

```yaml
nav:
  - Home: index.md
  - Results:
      - Overview: results/index.md
      - "2026-06-16": results/2026-06-16.md
```

Add generated build output to `.gitignore`:

```text
site/
```

Validate with:

```bash
python -m mkdocs build --strict
```

Run the local docs server with:

```bash
python -m mkdocs serve
```

The site should be reachable on the same laptop at:

```text
http://127.0.0.1:8000/
```

## Result Entry Format

Each result entry should look like this:

```markdown
## Short result title

![Short alt text](../assets/results/YYYY-MM-DD/figure_name.png)

*Caption describing the result and the important condition/model/data shown.*

Source: `relative/path/to/script.py`
Figure: `docs/assets/results/YYYY-MM-DD/figure_name.png`
```

## Skill to Create

Create a Codex skill named:

```text
update-result-book
```

Install it in the user's Codex skills folder, usually:

```text
~/.codex/skills/update-result-book/
```

The skill should trigger when the user asks to add, log, document, publish, or update a result figure in the result docs/book.

The skill instructions should tell future agents to:

1. Identify the generated figure path.
2. Identify the Python file or notebook that generated it.
3. Use the local date unless the user specifies another date.
4. Copy the figure into `docs/assets/results/YYYY-MM-DD/`.
5. Add an entry to `docs/results/YYYY-MM-DD.md`.
6. Update `docs/results/index.md`.
7. Update `mkdocs.yml` navigation with the quoted date label.
8. Run `python -m mkdocs build --strict`.
9. Report the page path and figure path back to the user.

If useful, add a helper script inside the skill:

```text
~/.codex/skills/update-result-book/scripts/update_result_book.py
```

The helper script should accept:

```bash
python ~/.codex/skills/update-result-book/scripts/update_result_book.py \
  --repo /path/to/repo \
  --figure path/to/generated.png \
  --source path/to/script.py \
  --title "Short result title" \
  --caption "Caption for the figure." \
  --date YYYY-MM-DD
```

The script should:

- copy the figure into `docs/assets/results/YYYY-MM-DD/`
- avoid overwriting by adding a numeric suffix when needed
- create the date page if it does not exist
- append the result entry
- update `docs/results/index.md`
- update `mkdocs.yml`

After creating the skill, validate it with the Codex skill validator if available.

## Important Details

- Keep the setup simple. This is for exploratory research notes, not production documentation.
- Do not over-engineer the docs structure.
- Prefer readable Markdown and simple paths.
- Always include the source script or notebook path with each figure.
- Always run the MkDocs strict build after changes.
- If a server is started, tell the user the exact local URL and how to stop/restart it.
