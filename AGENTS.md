# AGENTS.md

## Project Overview

`dtst` is a Python CLI toolbox for building and curating image datasets. It is structured as a main `dtst` command with subcommands for each pipeline stage. The package is managed with `uv` and installed via `pyproject.toml`.

## CLI Conventions

All commands use **Click** for argument parsing. Never use `argparse`.

### Command Structure

Every subcommand is a function decorated with `@click.command()` and registered to the main `dtst` CLI group. Commands live in `dtst/commands/` as individual modules.

```python
import click

@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Output directory")
@click.option("--workers", "-w", default=None, type=int, help="Number of parallel workers (default: CPU count)")
@click.option("--dry-run", is_flag=True, help="Preview what would be done without executing")
def my_command(input_dir, output, workers, dry_run):
    """One-line description of what this command does."""
    if workers is None:
        workers = cpu_count()
    # ...
```

### Option Patterns

- Always provide a short flag alias for frequently used options (`-o`, `-w`, `-t`)
- Use `click.Path(path_type=Path)` so paths arrive as `Path` objects
- Use `click.Path(exists=True)` for inputs that must already exist
- Default `--workers` to `None` and resolve to `cpu_count()` inside the function body
- Use `show_default=True` on options where the default value is meaningful
- Use `click.Choice([...])` for options with a fixed set of valid values

### Output Style

- Print a brief header summarizing the operation before starting work
- Use `click.echo()` for all output, never bare `print()`
- Write errors to stderr with `click.echo(..., err=True)`
- End with a summary line showing counts (processed, skipped, failed)

## Progress Reporting

All commands that iterate over files or batches use **tqdm** for progress bars.

```python
from tqdm import tqdm

with tqdm(total=len(items), desc="Processing images", unit="image") as pbar:
    for result in pool.imap(worker_fn, items):
        # handle result
        pbar.update(1)
```

- Always set `desc` to a short human-readable label
- Always set `unit` to what is being counted (image, url, page, file)
- Use `pbar.set_postfix()` to show live counters like matches or errors

## Parallelism

Use **`concurrent.futures`** as the unified parallelism API. The executor type depends on the workload. Every command that accepts `--workers` defaults to `cpu_count()` when not specified.

### Network I/O (search, fetch, URL downloading)

Use **`ThreadPoolExecutor`**. These tasks wait on HTTP responses, not CPU. Threads share memory, have near-zero startup cost, and the GIL does not matter because the bottleneck is network latency.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

def download_url(args):
    url, output_path = args
    # do network work
    return result

work = [(url, path) for url, path in items]
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(download_url, w): w for w in work}
    with tqdm(total=len(work), desc="Downloading", unit="url") as pbar:
        for future in as_completed(futures):
            result = future.result()
            # handle result
            pbar.update(1)
```

### CPU-bound processing (face detection, alignment, blur scoring, image hashing)

Use **`ProcessPoolExecutor`**. These tasks need true parallel CPU execution to escape the GIL. Each worker runs in its own process with its own Python interpreter.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def process_image(args):
    """Must be a top-level module function, not a lambda or closure."""
    image_path, config_value = args
    # do CPU work
    return result

work = [(path, config) for path in image_paths]
with ProcessPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(process_image, w): w for w in work}
    with tqdm(total=len(work), desc="Processing", unit="image") as pbar:
        for future in as_completed(futures):
            result = future.result()
            # handle result
            pbar.update(1)
```

- Worker functions must be **top-level module functions** (not lambdas, not closures, not methods)
- Pack all arguments into a single tuple and unpack inside the worker
- Workers must catch all exceptions internally and return error status, never let exceptions propagate

### GPU-bound inference (CLIP, IQA, face embeddings)

Use **single-process batched inference**. Never spawn multiple processes that each load a GPU model — this duplicates the model in VRAM and will crash on consumer GPUs.

Load the model once, then process images in batches:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = load_model(device)

for batch_paths in batched(image_paths, batch_size=32):
    images = torch.stack([preprocess(load(p)) for p in batch_paths]).to(device)
    with torch.no_grad():
        results = model(images)
    # store results per image
```

To maximize throughput, use a **ThreadPoolExecutor to preload and preprocess images** while the GPU is busy with the current batch:

```python
from concurrent.futures import ThreadPoolExecutor

def load_and_preprocess(path):
    image = Image.open(path).convert("RGB")
    return preprocess(image)

with ThreadPoolExecutor(max_workers=4) as loader:
    for batch_paths in batched(image_paths, batch_size=32):
        tensors = list(loader.map(load_and_preprocess, batch_paths))
        images = torch.stack(tensors).to(device)
        with torch.no_grad():
            results = model(images)
        # store results
```

### Choosing the right executor

| Workload | Bottleneck | Executor | Examples |
|----------|-----------|----------|----------|
| Network requests | I/O latency | `ThreadPoolExecutor` | search, fetch, URL validation |
| Image processing | CPU | `ProcessPoolExecutor` | face alignment, blur scoring, hashing, dedup |
| Model inference | GPU | Single process, batched | CLIP scoring, IQA metrics, face embeddings |

## File Conventions

- Use `pathlib.Path` everywhere, never string path manipulation
- Image extensions recognized: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`
- Provide a shared utility for finding image files in a directory:

```python
def find_images(directory: Path, recursive: bool = False) -> list[Path]:
```

- Metadata is stored as `metadata.json` in the dataset directory
- Subject configuration is a YAML file loaded with `pyyaml`
- Environment variables are loaded from `.env` with `python-dotenv`

## Logging

Use Python's **`logging`** module for all output. Never use bare `print()` or `click.echo()` for operational messages. The only exception is the final result summary at the end of a command, which is command output (not a log) and uses `click.echo()`.

### Logger Setup

Each module gets its own logger:

```python
import logging

logger = logging.getLogger(__name__)
```

The main CLI entry point configures the root logger based on verbosity:

```python
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose):
    """dtst - dataset toolkit for datasets creation and curation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
```

Every command wraps its main loop with `logging_redirect_tqdm()` so log lines don't collide with progress bars:

```python
from tqdm.contrib.logging import logging_redirect_tqdm

with logging_redirect_tqdm():
    with tqdm(total=len(items), desc="Processing", unit="image") as pbar:
        for result in process(items):
            # handle result
            pbar.update(1)
```

### Log Levels

| Level | Use for | Examples |
|-------|---------|----------|
| `DEBUG` | Per-item detail, only visible with `-v` | File processed, API response body, similarity score for one image, cache hit/miss |
| `INFO` | Operational milestones the user cares about | Step starting, item counts found, config loaded, model loaded |
| `WARNING` | Recoverable issues that don't stop the pipeline | Rate limit hit and retrying, file skipped as unreadable, API returned empty page, threshold looks unusual |
| `ERROR` | Individual item failures that don't halt the command | Single download failed, face detection crashed on one image, corrupt file |

### Rules

- Never log inside worker functions that run in `ProcessPoolExecutor` — logging across process boundaries is unreliable. Workers should return error information in their result and the main process logs it.
- `ThreadPoolExecutor` workers can log normally since threads share the same logging configuration.
- GPU inference loops should log batch progress at `DEBUG` level and only log errors at `ERROR` level.
- Every command supports `--verbose` / `-v` inherited from the top-level `dtst` group. No command defines its own verbosity flag.
- Suppress noisy third-party loggers in the CLI entry point as needed:

```python
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
```

## Error Handling

- Commands should not abort on individual item failures — log the error and continue
- Collect error counts and report in the final summary
- Use `click.ClickException` for fatal errors that should halt the command