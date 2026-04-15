"""Pre-generate docs/reference/cli.md so CI only needs zensical.

Usage: uv run scripts/gen_cli_docs.py
"""

from pathlib import Path

from mkdocs_click._docs import make_command_docs

from dtst.cli import cli

out = Path(__file__).resolve().parent.parent / "docs" / "reference" / "cli.md"
out.parent.mkdir(parents=True, exist_ok=True)
body = "\n".join(make_command_docs(prog_name="dtst", command=cli, depth=1, style="table", has_attr_list=True))
out.write_text("# CLI reference\n\n" + body + "\n")
print(f"Generated {out}")
