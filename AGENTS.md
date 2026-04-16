# AGENTS.md

## Project Overview

`dtst` is a Python CLI toolbox for building and curating image datasets. It is structured as a main `dtst` command with subcommands for each pipeline stage. The package is managed with `uv` and installed via `pyproject.toml`.

## Quick Start

```bash
uv sync                           # install dev environment
cp .env.example .env              # fill in API keys (see Environment)
uv run dtst --help                # verify install
```

The `dtst` console script is defined in `pyproject.toml`.

## Project Layout

- **`/dtst`** — Main package. CLI wrappers in `/dtst/cli/commands/`, core logic in `/dtst/core/`.
- **`/docs`** — User-facing documentation (Markdown); built with Zensical; `zensical.toml` at repo root.
- **`/tests`** — Pytest suite (see **Testing**).
- **`/scripts`** — Maintenance scripts, e.g. `gen_cli_docs.py`.

## Environment

Runtime reads `.env` via `python-dotenv`. Keys (see `.env.example`):

- `BRAVE_API_KEY`, `FLICKR_API_KEY`, `SERPER_API_KEY` — search engine credentials
- `DTST_USER_AGENT` — HTTP User-Agent string for outbound requests

## CLI Conventions

All commands use **Click** for argument parsing. Never use `argparse`.

### Pipeline Model

The pipeline is organized around a **working directory** that acts as the project root. Users can inspect it in the file explorer, drop files in, and everything just works.

`--working-dir` / `-d` is equivalent to `cd` into that directory before the command runs, and is optional. Path options (`--from`, `--to`, `--input`, `--output`) accept any relative or absolute path — they resolve against cwd like in any other tool.

There are three kinds of commands:

- **Sourcing** commands bring images in from the outside world (e.g. `fetch`). They have a `--to` option but no `--from`.
- **Augmenting** commands take one or more existing folders and produce a new folder (e.g. `extract-faces`). They have both `--from` and `--to`.
- **Filtering** commands reduce an existing folder without producing a new one.

`search` is a special case: it has no `--from` or `--to` and simply writes `results.jsonl` for `fetch` to consume.

The conventional happy-path pipeline uses these folder names under the working directory:

```
working_dir/
  results.jsonl       ← search output
  raw/                ← fetch output (--to raw)
  extra/              ← manually added images
  faces/              ← extract-faces output (--to faces, --from raw)
```

### Command Structure

Every subcommand is a `@click.command()` registered on the main `dtst` group. The split is:

- **CLI wrapper** in `dtst/cli/commands/<name>.py` — Click decorators (`@config_argument`, `@working_dir_option()`, `@click.option(...)`), validation, then `apply_working_dir(working_dir)` and a lazy import of the core function. Core functions do not accept `working_dir`.
- **Core function** in `dtst/core/<name>.py` — pure logic, no Click.

Copy any existing wrapper (e.g. `dtst/cli/commands/fetch.py`) as a template.

After changing any command's options, arguments, or docstring, regenerate the CLI reference:

```bash
uv run scripts/gen_cli_docs.py
```

### Configuration and CLI Override

Commands can be invoked with just a config file, just CLI options, or both. When both are provided, CLI options override config file values.

This uses Click's `default_map` mechanism (same pattern as Black). The `@config_argument` decorator adds an optional YAML config positional argument with an eager callback. When a config file is passed, the callback parses the YAML, extracts the command's section, and injects values into `ctx.default_map`. Click then uses these as defaults that CLI flags override automatically.

The schema is defined **once** — in the Click decorators. There are no config dataclasses or manual merge logic. YAML keys must match Click parameter names, with mappings for the four known mismatches handled in `_YAML_TO_CLICK` in `dtst/config.py`:

| YAML key | Click parameter | Reason |
|----------|----------------|--------|
| `from` | `from_dirs` | `from` is a Python keyword |
| `format` | `fmt` | `format` is a Python builtin |
| `input` | `input_file` | `input` is a Python builtin |
| `license` | `license_filter` | Avoids shadowing the builtin |

Config files use YAML with an optional `working_dir` at the top level and parameters nested under command-specific keys. When set, `working_dir` is resolved relative to the YAML file's directory and the command `chdir`s into it before running.

```yaml
working_dir: "./scratch/chanterelle"

fetch:
  to: raw
  min_size: 512

extract_faces:
  from:
    - raw
    - extra
  to: faces
```

YAML values are coerced to what Click expects by `_coerce_for_click()` in `dtst/config.py`: lists become comma-separated strings for `type=str` options, dicts become tuples-of-tuples for `type=(str, float), multiple=True` options, etc.

### Required Fields

Since `default_map` values act as defaults, required fields cannot use Click's `required=True` (which would fail when no config is provided). Instead, validate required fields explicitly in the command body:

```python
if from_dirs is None:
    raise click.ClickException("--from is required (or set 'my_command.from' in config)")
```

### Comma-Separated List Options

Options like `--from` accept comma-separated strings from the CLI. YAML lists are coerced to comma strings by the config callback. Commands split them inline:

```python
dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
```

### Option Patterns

- Always provide a short flag alias for frequently used options (`-d`, `-w`, `-t`)
- Use `--working-dir` / `-d` on all pipeline commands
- Use `--from` for augmenting commands that accept input folders (comma-separated, supports globs like `images/*`, maps to `from_dirs` in Python since `from` is a keyword). Glob expansion is handled by `resolve_dirs()` in `dtst/files.py`.
- Use `--to` for sourcing and augmenting commands that write to an output folder
- Use `click.Path(path_type=Path)` so paths arrive as `Path` objects
- Default `--workers` to `None` and resolve to `cpu_count()` inside the function body
- Default options that have config-file defaults to `None` so `default_map` values can take effect. Apply fallback defaults in the command body (e.g. `quality = quality if quality is not None else 95`).
- Use `show_default=True` on options where the default value is meaningful
- Use `click.Choice([...])` for enum validation — this replaces manual checks and works with both CLI and config values
- Use `click.IntRange(min, max)` or `click.FloatRange(min, max)` for range validation

## Progress Reporting

All commands that iterate over files or batches use **tqdm**. Always set `desc` (short human-readable label) and `unit` (what's being counted: `image`, `url`, `page`, `file`). Use `pbar.set_postfix()` for live counters like matches or errors.

## Parallelism

Use **`concurrent.futures`** as the unified parallelism API. Pick the executor by workload bottleneck. Commands that accept `--workers` default to `cpu_count()` when unspecified.

| Workload | Bottleneck | Executor | Examples |
|----------|-----------|----------|----------|
| Network requests | I/O latency | `ThreadPoolExecutor` | search, fetch, URL validation |
| Image processing | CPU | `ProcessPoolExecutor` | face alignment, blur scoring, hashing, dedup |
| Model inference | GPU | Single process, batched | CLIP scoring, IQA metrics, face embeddings |

**ProcessPoolExecutor rules:**

- Worker functions must be **top-level module functions** (not lambdas, closures, or methods) so they can pickle.
- Pack all arguments into a single tuple and unpack inside the worker.
- Workers must catch all exceptions internally and return an error status — never let exceptions propagate across process boundaries.

**GPU inference — critical:** Never spawn multiple processes that each load a GPU model. This duplicates the model in VRAM and crashes consumer GPUs. Load the model **once** in the main process and iterate in batches. To maximize throughput, use a small `ThreadPoolExecutor` to preload/preprocess the next batch on CPU while the GPU works on the current one. Device selection: `cuda` → `mps` → `cpu`.

## File Conventions

- Use `pathlib.Path` everywhere, never string path manipulation
- Image extensions recognized: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`
- Provide a shared utility for finding image files in a directory:

```python
def find_images(directory: Path, recursive: bool = False) -> list[Path]:
```

- Metadata is stored as `metadata.json` in the dataset directory
- Command configuration is a YAML file loaded via `@config_argument` (see **Configuration and CLI Override**). YAML sections are keyed by command name with underscores (`search:`, `fetch:`, `extract_faces:`, etc.)

## Logging & Output

Use Python's **`logging`** module for all operational output. Each module does `logger = logging.getLogger(__name__)`. The root logger is configured once in `dtst/cli/__init__.py` based on the top-level `--verbose` / `-v` flag — no command defines its own verbosity flag. Suppress noisy third-party loggers (`urllib3`, `PIL`, …) there as needed. Wrap any tqdm loop with `logging_redirect_tqdm()` so log lines don't collide with the progress bar.

The **final summary line** at the end of a command is command output, not a log — use `click.echo()` for it (and `click.echo(..., err=True)` for fatal messages). Never use bare `print()`.

### Log Levels

| Level | Use for | Examples |
|-------|---------|----------|
| `DEBUG` | Per-item detail, only visible with `-v` | File processed, API response body, per-image similarity score, cache hit/miss |
| `INFO` | Operational milestones | Step starting, item counts found, config loaded, model loaded |
| `WARNING` | Recoverable issues | Rate limit + retry, file skipped as unreadable, empty API page, unusual threshold |
| `ERROR` | Individual item failures that don't halt the command | Single download failed, detection crashed on one image, corrupt file |

### Rules

- Never log inside `ProcessPoolExecutor` workers — logging across process boundaries is unreliable. Return error info in the result and let the main process log it.
- `ThreadPoolExecutor` workers can log normally (shared config).
- GPU inference loops: batch progress at `DEBUG`, errors at `ERROR`.

## Code Quality

Use **ruff** for formatting and linting. Run both before every commit:

```bash
uv run ruff format dtst/
uv run ruff check --fix dtst/
```

## Testing

Tests live in `/tests` and are run with **pytest** via uv:

```bash
uv run pytest tests/ -q              # run suite
uv run pytest tests/ --cov           # with coverage
```

### Scope

Unit-test pure-logic helpers, CLI wiring, and pure bits carved out of `core`/`engines` (HTTP engines are tested with `responses`-mocked requests). Do **not** unit-test heavy ML inference modules — `dtst/embeddings/`, `dtst/detections/`, `dtst/metrics/`, and the pipeline orchestrators in `dtst/core/` — they are validated by manual pipeline runs. Existing tests are characterization-style: when changing behavior intentionally, flip the assertion rather than deleting the test.

### CLI lazy-import rule (critical)

Every `dtst/cli/commands/*.py` wrapper MUST import its core module **inside the `cmd()` function body**, not at module top-level. Pattern:

```python
def cmd(working_dir, ...):
    apply_working_dir(working_dir)
    # validation here
    from dtst.core.fetch import fetch as core_fetch   # lazy
    return core_fetch(...)
```

This keeps `dtst --help` under 200ms and CI deps lean. The regression test `tests/test_cli.py::test_cli_import_does_not_load_torch` enforces this by asserting that `import dtst.cli` does not transitively load torch, transformers, PIL, insightface, mediapipe, dlib, onnxruntime, open_clip, pyiqa, or spandrel. `dtst/__init__.py` uses PEP 562 `__getattr__` for the same reason — do not re-add eager `dtst.core.*` imports there.

### CI

`.github/workflows/tests.yml` installs a lean runtime set plus CPU-only torch — NOT the full ML stack. If a new test requires a runtime dep not currently installed in CI, either keep the import lazy or add the dep to the `Install lean test + runtime deps` step.

## Error Handling

- Commands should not abort on individual item failures — log the error and continue
- Collect error counts and report in the final summary
- Use `click.ClickException` for fatal errors that should halt the command

## Documentation

See **Project layout** above for the documentation path (`/docs`). Documentation lives in `/docs` as Markdown and is built with **Zensical**. The project uses an `zensical.toml` configuration file.

### CLI Reference (pre-generated)

CLI command documentation is pre-generated from Click source code by `scripts/gen_cli_docs.py`. The script introspects the `dtst` CLI group and emits `docs/reference/cli.md`. This means the Click decorators and docstrings are the single source of truth for the CLI reference.

After changing any command's options, arguments, or docstring, regenerate the CLI docs:

```bash
uv run scripts/gen_cli_docs.py
```

Because Click decorators are the source of truth for the CLI reference, every command needs a clear one-line docstring, every option needs `help=`, and options with meaningful defaults use `show_default=True`. Usage examples go under a `\b` escape in the docstring so Click doesn't rewrap them.

### Hand-written documentation

Conceptual docs (getting started guide, pipeline overview, tutorials, architecture explanation) are written by hand in `/docs`. These are standard Markdown files organized as:

```
docs/
  index.md              # Project overview and quick start
  getting-started.md    # Installation and first run
  pipeline.md           # Pipeline stages explained
  configuration.md      # Subject YAML config reference
  reference/
    cli.md              # Pre-generated CLI reference (scripts/gen_cli_docs.py)
```

Keep hand-written docs focused on the *why* and *how* — the CLI reference handles the *what*.