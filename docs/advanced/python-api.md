# Python API

`dtst` is usable as a library, not just a CLI. Every pipeline command has a matching function in the `dtst` package that you can call from Python. This is useful for embedding dtst in a larger data pipeline, driving it from a notebook, or composing commands in ways the CLI does not express directly.

## Quick start

```python
from pathlib import Path
from dtst import fetch, extract_frames, detect, extract_classes, frame

WORKING_DIR = Path("./scratch/trees")

fetch(
    working_dir=WORKING_DIR,
    to="raw",
    input_file="urls.txt",
)

extract_frames(
    working_dir=WORKING_DIR,
    from_dirs="raw",
    to="frames",
    keyframes=10.0,
)

detect(
    working_dir=WORKING_DIR,
    from_dirs="frames",
    classes="tree",
)

extract_classes(
    working_dir=WORKING_DIR,
    from_dirs="frames",
    to="crops",
    classes="tree",
    square=True,
)

frame(
    working_dir=WORKING_DIR,
    from_dirs="crops",
    to="framed",
    width=256,
    height=256,
    mode="stretch",
)
```

## Function signatures

Every command is exported at the top level of `dtst`:

```python
from dtst import (
    analyze, annotate, augment, cluster, dedup, detect,
    extract_classes, extract_faces, extract_frames,
    fetch, format, frame, rename, search, select,
    upscale, validate,
)
```

Each function takes **keyword-only arguments** that mirror the CLI flags. Parameter names match the long flag, with `--from` → `from_dirs` and `--format` → `fmt` (the two cases where the CLI name collides with a Python keyword or builtin). Required fields raise `InputError` if missing.

The `review` and `run` commands are CLI-only and not available as library functions.

## Return values

Each function returns a dataclass describing what happened. For example, `validate`:

```python
from dtst import validate

result = validate(
    working_dir=Path("./my-dataset"),
    from_dirs="faces",
    square=True,
)
if not result.passed:
    print(f"Failed: {result.failed} errors, {result.non_square} non-square")
```

Every result dataclass is re-exported from `dtst`: `FetchResult`, `DetectResult`, `ExtractClassesResult`, `FrameResult`, `ValidateResult`, `ClusterResult`, etc. See [`dtst/results.py`](https://github.com/Ucodia/dtst/blob/main/dtst/results.py) for the full list.

## Error handling

Library calls raise subclasses of `DtstError` instead of exiting the process:

```python
from dtst import fetch, DtstError, InputError, PipelineError

try:
    fetch(working_dir=working, to="raw", input_file="urls.txt")
except InputError as e:
    # Bad / missing inputs (e.g. no URLs to fetch, unsupported file format)
    ...
except PipelineError as e:
    # Pipeline-level failure (e.g. no images produced valid embeddings)
    ...
except DtstError as e:
    # Base class catches all library-layer errors
    ...
```

Individual per-item failures (a single unreadable image, a single failed download) do **not** raise — they are logged and counted in the result.

## Silencing progress bars

Every function that shows a `tqdm` progress bar accepts `progress: bool = True`. Pass `progress=False` to silence output when embedding dtst in a larger program or running in a notebook that renders its own UI:

```python
result = detect(
    working_dir=working,
    from_dirs="frames",
    classes="tree",
    progress=False,
)
```

Logs still go through Python's standard `logging` module. Configure the root logger to control verbosity:

```python
import logging
logging.basicConfig(level=logging.WARNING)  # quiet
logging.basicConfig(level=logging.DEBUG)    # verbose
```

## Async usage

`dtst.aio` mirrors the sync API as `async def` wrappers built on `asyncio.to_thread`. Use it when you want to call dtst from an async program without blocking the event loop:

```python
from dtst.aio import fetch, detect, frame

result = await fetch(
    working_dir=Path("./scratch/trees"),
    to="raw",
    input_file="urls.txt",
)
```

There is no native async I/O underneath — dtst's workloads are CPU/GPU-bound or already use thread pools internally, so `to_thread` is the right abstraction. Everything else (arguments, return values, exceptions, `progress=False`) behaves identically to the sync API.

## Library vs. CLI layer

`dtst` is organized into two layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| Core library | `dtst.core` | Pure Python functions; no Click dependency. Accepts typed arguments, returns dataclasses, raises `DtstError`. |
| CLI | `dtst.cli` | Thin Click wrappers around `dtst.core`. Parses flags, resolves the YAML config, formats output for the terminal, maps exceptions to process exit codes. |

Top-level `dtst.<command>` imports re-export the core functions, so `from dtst import validate` and `from dtst.core import validate` are equivalent. `dtst.cli` is only needed if you want to invoke the CLI group programmatically.

## YAML config files

The `@config_argument` decorator and YAML loading live in the CLI layer and are not applied when calling core functions directly. If you want to reuse an existing YAML pipeline config from Python, parse it yourself and forward the values:

```python
import yaml
from pathlib import Path
from dtst import detect

with open("tree.yaml") as f:
    cfg = yaml.safe_load(f)

detect(
    working_dir=Path(cfg["working_dir"]),
    from_dirs=cfg["detect"]["from"],
    classes=cfg["detect"]["classes"],
)
```

For replayable multi-step workflows, prefer `dtst run` — see [Workflows](workflows.md).
