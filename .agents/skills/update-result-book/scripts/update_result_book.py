#!/usr/bin/env python3
import argparse
import datetime as dt
import shutil
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Add a generated figure to the date-wise MkDocs result book."
)
parser.add_argument("--repo", default=".", help="Repository root containing mkdocs.yml")
parser.add_argument("--figure", required=True, help="Generated figure path")
parser.add_argument("--source", required=True, help="Python/notebook path that generated the figure")
parser.add_argument("--title", required=True, help="Result entry title")
parser.add_argument("--caption", required=True, help="Figure caption")
parser.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD result page date")
args = parser.parse_args()

repo = Path(args.repo).resolve()
docs_dir = repo / "docs"
results_dir = docs_dir / "results"
assets_dir = docs_dir / "assets" / "results" / args.date
mkdocs_path = repo / "mkdocs.yml"
results_index_path = results_dir / "index.md"
date_page_path = results_dir / f"{args.date}.md"

figure_path = Path(args.figure).expanduser()
if not figure_path.is_absolute():
    figure_path = (repo / figure_path).resolve()
else:
    figure_path = figure_path.resolve()

source_path = Path(args.source).expanduser()
if not source_path.is_absolute():
    source_path = (repo / source_path).resolve()
else:
    source_path = source_path.resolve()

if not mkdocs_path.exists():
    raise SystemExit(f"Missing MkDocs config: {mkdocs_path}")
if not figure_path.exists():
    raise SystemExit(f"Missing figure: {figure_path}")
if not source_path.exists():
    raise SystemExit(f"Missing source file: {source_path}")

assets_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

dest_figure = assets_dir / figure_path.name
if dest_figure.exists():
    stem = figure_path.stem
    suffix = figure_path.suffix
    counter = 2
    while True:
        candidate = assets_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            dest_figure = candidate
            break
        counter += 1
shutil.copy2(figure_path, dest_figure)

try:
    source_ref = source_path.relative_to(repo)
except ValueError:
    source_ref = source_path

figure_ref = dest_figure.relative_to(repo)
page_figure_ref = Path("..") / "assets" / "results" / args.date / dest_figure.name

if not date_page_path.exists():
    date_page_path.write_text(
        f"# Results: {args.date}\n\nAdd result entries below this line.\n",
        encoding="utf-8",
    )

entry = (
    f"\n## {args.title}\n\n"
    f"![{args.caption}]({page_figure_ref.as_posix()})\n\n"
    f"*{args.caption}*\n\n"
    f"Source: `{source_ref.as_posix()}`\n"
    f"Figure: `{figure_ref.as_posix()}`\n"
)
with date_page_path.open("a", encoding="utf-8") as f:
    f.write(entry)

index_link = f"- [{args.date}]({args.date}.md)"
if results_index_path.exists():
    index_text = results_index_path.read_text(encoding="utf-8")
else:
    index_text = "# Results\n\nDaily result pages:\n"
if index_link not in index_text:
    if not index_text.endswith("\n"):
        index_text += "\n"
    index_text += index_link + "\n"
    results_index_path.write_text(index_text, encoding="utf-8")

nav_line = f'      - "{args.date}": results/{args.date}.md'
mkdocs_text = mkdocs_path.read_text(encoding="utf-8")
if nav_line not in mkdocs_text:
    lines = mkdocs_text.splitlines()
    insert_at = None
    for i, line in enumerate(lines):
        if line == "  - Results:":
            insert_at = i + 1
            while insert_at < len(lines) and (
                lines[insert_at].startswith("      ") or lines[insert_at].strip() == ""
            ):
                insert_at += 1
            break
    if insert_at is None:
        for i, line in enumerate(lines):
            if line == "nav:":
                insert_at = i + 1
                lines.insert(insert_at, "  - Results:")
                insert_at += 1
                lines.insert(insert_at, "      - Overview: results/index.md")
                insert_at += 1
                break
    if insert_at is None:
        lines.extend(["", "nav:", "  - Results:", "      - Overview: results/index.md"])
        insert_at = len(lines)
    lines.insert(insert_at, nav_line)
    mkdocs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Updated {date_page_path}")
print(f"Copied figure to {dest_figure}")
