# Concepts

## Buckets

Every `dtst` command reads from and writes to **buckets** -- plain folders on disk. There is nothing to register or configure: drop files into a folder and it becomes a bucket. You can browse buckets in the file explorer, manually add or remove files, and the tools will pick up whatever is there.

Commands reference buckets through `--from` and `--to` flags:

- **Sourcing** commands like `fetch` only have `--to` -- they bring data into a bucket from the outside world.
- **Augmenting** commands like `extract-faces` or `frame` have both `--from` and `--to` -- they read from one or more buckets and write results to another.
- **Filtering** commands like `dedup` or `review` operate on a bucket in place.

The `--from` flag accepts comma-separated names and supports globs. For example `--from images/*` matches all subdirectories of `images/`.

Because buckets are plain directories, you can always supplement pipeline output by manually copying files into a bucket. No import step is needed.

## Sidecars

A **sidecar** is a small JSON file that lives next to an image or video and carries metadata about it. The naming convention appends `.json` to the full filename:

```
photo.jpg
photo.jpg.json
```

### What goes in a sidecar

Different commands write different fields:

| Command | Fields | Description |
|---------|--------|-------------|
| `fetch` | `source`, `origin`, `license` | Where the file came from, its original URL, and license info |
| `analyze --metrics phash` | `metrics.phash` | Perceptual hash for duplicate detection |
| `analyze --metrics blur` | `metrics.blur` | Laplacian variance measuring image sharpness |
| `analyze --metrics <iqa>` | `metrics.<iqa>` | IQA-PyTorch quality scores (e.g. musiq, clipiqa) |
| `detect` | `classes` | Object detection results (score + bounding box per class) |

A sidecar after fetching, detecting and analyzing might look like this:

```json
{
  "source": "flickr",
  "origin": "https://live.staticflickr.com/example/photo.jpg",
  "license": "cc-by",
  "metrics": {
    "phash": "d4c6b8b0b4e1e3c7",
    "blur": 1842.57
  },
  "classes": {
    "microphone": [],
    "chair": [
      { "score": 0.87, "box": [10, 200, 250, 480] }
    ]
  }
}
```

### How sidecars flow through the pipeline

Sidecars are automatically carried along when images move between buckets. The exact behavior depends on whether the command transforms the image pixels:

**Commands that copy images unchanged** (cluster, select, dedup, review) preserve the full sidecar as-is. Every field carries over because nothing about the image has changed.

**Commands that transform images** (augment, extract-faces, extract-frames, frame, upscale) carry over provenance and semantic fields (`source`, `origin`, `license`, `tags`) but drop computed fields (`metrics`, `classes`). These pixel-dependent values would be invalid after the transformation and need to be recomputed with `analyze` or `detect` if needed.
