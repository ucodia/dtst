# Python API

`dtst` is usable as a library, not just a CLI. Every pipeline command has a matching function in the `dtst` package that you can call from Python. This is useful for embedding dtst in a larger data pipeline, driving it from a notebook, or composing commands in ways the CLI does not express directly.

## Quick start

```python
from pathlib import Path
from dtst import fetch, extract_frames, detect, extract_classes, frame

base = Path("./scratch/trees")

fetch(to=str(base / "raw"), input_file=str(base / "urls.txt"))

extract_frames(
    from_dirs=str(base / "raw"),
    to=str(base / "frames"),
    keyframes=10.0,
)

detect(from_dirs=str(base / "frames"), classes="tree")

extract_classes(
    from_dirs=str(base / "frames"),
    to=str(base / "crops"),
    classes="tree",
    square=True,
)

frame(
    from_dirs=str(base / "crops"),
    to=str(base / "framed"),
    width=256,
    height=256,
    mode="stretch",
)
```

Path arguments (`from_dirs`, `to`, `input_file`, `output`) resolve against the current working directory when relative, or are used as-is when absolute. In scripts, build paths from a shared base directory rather than relying on cwd.

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

Core functions do not accept `working_dir`; it is a CLI convenience that `chdir`s before the command runs. From Python, construct paths from a base directory (as in the quick start) and pass them in directly.

The `review` and `run` commands are CLI-only and not available as library functions.

## Return values

Each function returns a dataclass describing what happened. For example, `validate`:

```python
from dtst import validate

result = validate(from_dirs="./my-dataset/faces", square=True)
if not result.passed:
    print(f"Failed: {result.failed} errors, {result.non_square} non-square")
```

Every result dataclass is re-exported from `dtst`: `FetchResult`, `DetectResult`, `ExtractClassesResult`, `FrameResult`, `ValidateResult`, `ClusterResult`, etc. See [`dtst/results.py`](https://github.com/Ucodia/dtst/blob/main/dtst/results.py) for the full list.

## Error handling

Library calls raise subclasses of `DtstError` instead of exiting the process:

```python
from dtst import fetch, DtstError, InputError, PipelineError

try:
    fetch(to="raw", input_file="urls.txt")
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
from pathlib import Path
from dtst.aio import fetch, detect, frame

base = Path("./scratch/trees")
result = await fetch(
    to=str(base / "raw"),
    input_file=str(base / "urls.txt"),
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
from pathlib import Path
import yaml
from dtst import detect

config_path = Path("tree.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

base = config_path.parent / cfg.get("working_dir", ".")

detect(
    from_dirs=str(base / cfg["detect"]["from"]),
    classes=cfg["detect"]["classes"],
)
```

For replayable multi-step workflows, prefer `dtst run` — see [Workflows](workflows.md).
