# dtst { #dtst data-toc-label='dtst' }

dtst - dataset toolkit for datasets creation and curation.

**Usage:**

```text
dtst [OPTIONS] COMMAND [ARGS]...
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--verbose`, `-v` | boolean | Enable debug logging | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst analyze { #dtst-analyze data-toc-label='analyze' }

Compute image metadata and write JSON sidecars.

Analyzes images in the source folders and writes per-image sidecar
JSON files containing the requested metadata (perceptual hash,
blur score, or both). Sidecars are merged incrementally — running
with --phash then --blur accumulates both.

At least one analyzer flag (--phash, --blur) is required unless
using --clear.

Examples:

  dtst analyze --from raw --phash --blur -d ./my-dataset
  dtst analyze config.yaml --phash
  dtst analyze --from raw,extra --blur --force
  dtst analyze --from raw --phash --dry-run -d ./my-dataset
  dtst analyze --from raw --clear -d ./my-dataset

**Usage:**

```text
dtst analyze [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Comma-separated source folders (supports globs, e.g. 'images/*'). | None |
| `--phash` | boolean | Compute perceptual hash for each image. | `False` |
| `--blur` | boolean | Compute blur score (Laplacian variance) for each image. | `False` |
| `--force` | boolean | Recompute all analyzers even if sidecar data already exists. | `False` |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--clear` | boolean | Remove all sidecar files from source folders. | `False` |
| `--dry-run` | boolean | Preview what would be computed without writing sidecars. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst augment { #dtst-augment data-toc-label='augment' }

Augment a dataset by applying image transformations.

Reads images from one or more source folders and writes transformed
copies to a destination folder. By default the original images are
also copied to the output; use --no-copy to write only the
transformed versions.

At least one transform flag (--flipX, --flipY, --flipXY) is
required. Multiple flags can be combined in a single run to
produce several variants of each image.

Transformed files are named with a suffix indicating the transform:
photo.jpg becomes photo_flipX.jpg, photo_flipY.jpg, photo_flipXY.jpg.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:

    dtst augment -d ./project --from faces --to augmented --flipX
    dtst augment -d ./project --from faces --to augmented --flipX --flipY --flipXY
    dtst augment -d ./project --from faces --to augmented --flipX --no-copy
    dtst augment config.yaml --dry-run

**Usage:**

```text
dtst augment [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--flipX` | boolean | Apply horizontal flip. | `False` |
| `--flipY` | boolean | Apply vertical flip. | `False` |
| `--flipXY` | boolean | Apply both horizontal and vertical flip (180-degree rotation). | `False` |
| `--no-copy` | boolean | Do not copy original images to the output folder. | `False` |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--dry-run` | boolean | Preview what would be written without creating files. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst cluster { #dtst-cluster data-toc-label='cluster' }

Cluster images by visual similarity.

Groups images into clusters based on embedding similarity using
HDBSCAN. Each cluster is written to a numbered subdirectory
(000 = largest, 001 = second largest, etc.) within the output
folder. Images that do not belong to any cluster are placed in
a noise/ subdirectory.

Supports two embedding models: arcface for face identity
clustering (requires face images) and clip for general visual
similarity clustering (works with any images).

--min-cluster-size sets the smallest group HDBSCAN will consider
a real cluster (default: 5). Raise it to suppress small or
spurious clusters; lower it to capture smaller groups.

--min-samples controls how conservative the density estimate is
(default: 2). It decides how many close neighbors a point needs
before it can join a cluster. Lower values (1-2) let borderline
images in; higher values push more images into the noise folder.
Keeping this low while adjusting --min-cluster-size is usually
the best starting point.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:

    dtst cluster config.yaml
    dtst cluster -d ./project --from faces --to clusters
    dtst cluster -d ./project --model clip --from raw --to clusters
    dtst cluster -d ./project --top 3 --min-cluster-size 10
    dtst cluster -d ./project --min-samples 1 --min-cluster-size 8
    dtst cluster config.yaml --model arcface --dry-run

**Usage:**

```text
dtst cluster [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to`, `-t` | text | Destination folder name within the working directory. | None |
| `--model`, `-m` | choice (`arcface` &#x7C; `clip`) | Embedding model for similarity (default: arcface). | None |
| `--top`, `-n` | integer | Maximum number of clusters to output; omit for all clusters. | None |
| `--min-cluster-size` | integer | Minimum images to form a cluster (default: 5). | None |
| `--min-samples` | integer | How many close neighbors a point needs to join a cluster; lower values include more borderline images (default: 2). | None |
| `--batch-size`, `-b` | integer | Images per inference batch (default: 32). | None |
| `--workers`, `-w` | integer | Number of workers for image preloading (default: CPU count). | None |
| `--no-cache` | boolean | Skip the embedding cache and recompute from scratch. | `False` |
| `--clean` | boolean | Remove the output directory before writing new clusters. | `False` |
| `--dry-run` | boolean | Show image count and configuration without clustering. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst copy { #dtst-copy data-toc-label='copy' }

Copy images from one or more folders to a destination folder.

Duplicates the contents of the source folders into the destination
without any transformation. Files that already exist in the
destination (by name) are skipped.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:
    dtst copy -d ./project --from raw --to backup
    dtst copy -d ./project --from raw,extra --to combined
    dtst copy config.yaml --dry-run

**Usage:**

```text
dtst copy [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--dry-run` | boolean | Preview what would be copied without creating files. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst dedup { #dtst-dedup data-toc-label='dedup' }

Deduplicate images by perceptual hash similarity.

Groups images by phash hamming distance and keeps the best image
from each duplicate group. The winner is chosen by resolution
(width x height), then file size, then blur sharpness. Losers are
moved to a duplicated/ subdirectory within the source folder
(configurable with --to).

Requires phash sidecar data from ``dtst analyze --phash``. Blur
scores (from ``dtst analyze --blur``) are used as a tiebreaker
when available.

Examples:
  dtst dedup -d ./project --from faces
  dtst dedup -d ./project --from faces --threshold 4
  dtst dedup -d ./project --from faces --to my-dupes
  dtst dedup config.yaml --dry-run
  dtst dedup -d ./project --from faces --clear

**Usage:**

```text
dtst dedup [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--from` | text | Folder name to deduplicate within the working directory. | None |
| `--to` | text | Subfolder name for duplicate images. | None |
| `--threshold`, `-t` | integer | Phash hamming distance threshold for near-duplicate detection. | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--clear` | boolean | Restore all deduplicated images back to the source folder. | `False` |
| `--dry-run` | boolean | Show what would be deduplicated without moving anything. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst extract-faces { #dtst-extract-faces data-toc-label='extract-faces' }

Extract aligned face crops from images.

Detects faces in each image using MediaPipe (default) or dlib,
then produces an aligned and cropped face image for each detection.
The alignment normalises eye and mouth positions for consistent
face crops.

Reads images from one or more source folders within the working
directory and writes face crops to a destination folder. Multiple
source folders can be specified as a comma-separated list with
--from.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:

    dtst extract-faces config.yaml
    dtst extract-faces config.yaml --engine dlib --max-size 512
    dtst extract-faces -d ./crowd --from raw --to faces
    dtst extract-faces -d ./crowd --from raw,extra --to faces
    dtst extract-faces config.yaml --max-faces 3 --no-padding

**Usage:**

```text
dtst extract-faces [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--max-size`, `-M` | integer | Maximum side length in pixels; faces smaller than this are kept at natural size (default: no limit). | None |
| `--engine`, `-e` | choice (`mediapipe` &#x7C; `dlib`) | Face detection engine (default: mediapipe). | None |
| `--max-faces`, `-m` | integer | Max faces to extract per image (default: 1). | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--padding` / `--no-padding` | boolean | Enable/disable reflective padding on crops (default: enabled). | None |
| `--skip-partial` | boolean | Skip faces whose crop extends beyond the image boundary instead of padding them. | `False` |
| `--refine-landmarks` | boolean | Enable MediaPipe refined landmarks (478 vs 468). | `False` |
| `--debug` | boolean | Overlay landmark points on output images. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst extract-frames { #dtst-extract-frames data-toc-label='extract-frames' }

Extract keyframes from video files using ffmpeg.

Reads video files from one or more source folders and extracts
keyframes (I-frames) to a destination folder. Each video produces
a set of numbered images named as
``{video_stem}_{frame_number}.{format}``.

Only I-frames are decoded, which avoids interpolated or blurry
frames and produces the sharpest possible output. The --keyframes
option sets the minimum interval between extracted frames: with
the default of 10, at most one keyframe every 10 seconds is kept.
Lower values produce more frames, higher values produce fewer.

Supported video formats: .mp4, .mkv, .avi, .mov, .webm, .flv,
.wmv, .m4v.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:

    dtst extract-frames -d ./project --from videos --to frames
    dtst extract-frames -d ./project --from videos --to frames --keyframes 5
    dtst extract-frames -d ./project --from videos --to frames --keyframes 30 --format png
    dtst extract-frames config.yaml
    dtst extract-frames config.yaml --keyframes 20 --dry-run

**Usage:**

```text
dtst extract-frames [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--keyframes`, `-k` | float | Minimum interval in seconds between extracted keyframes. Only I-frames are considered; frames closer together than this value are skipped (default: 10). | None |
| `--format`, `-F` | choice (`jpg` &#x7C; `png`) | Output image format (default: jpg). | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--dry-run` | boolean | Preview what would be done without extracting frames. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst fetch { #dtst-fetch data-toc-label='fetch' }

Download images and videos from a URL list.

Reads a URL list from the working directory specified by --input.
Two formats are supported:

  .jsonl  JSON Lines with a "url" field per line (search output).
          Supports --min-size and --license filtering.
  .txt    Plain text with one URL per line. Lines starting with
          # are treated as comments.

URLs are routed automatically: known video hosting domains
(YouTube, Vimeo, etc.) are downloaded with yt-dlp, all other
URLs are downloaded directly with HTTP requests.

Image files are named by the MD5 hash of the URL. Video files
are named by yt-dlp using the video ID and original extension.
Existing files are skipped unless --force is set.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:

    dtst fetch config.yaml
    dtst fetch -d ./chanterelle --to raw
    dtst fetch -d ./project --to videos --input urls.txt
    dtst fetch config.yaml --workers 16 --timeout 60
    dtst fetch config.yaml --force
    dtst fetch -d ./chanterelle --to raw --no-wait --license cc

**Usage:**

```text
dtst fetch [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory where input is read from and media is written to (default: .). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--input`, `-i` | text | Input file name relative to the working directory. Supports .jsonl and .txt formats. | None |
| `--min-size`, `-s` | integer | Minimum image dimension in pixels; only applies to .jsonl input (default: 512). | None |
| `--workers`, `-w` | integer | Number of parallel download threads (default: CPU count for images, 2 for video). | None |
| `--timeout`, `-t` | integer | Per-request timeout in seconds. | `30` |
| `--force`, `-f` | boolean | Re-download files even if they already exist. | `False` |
| `--max-wait`, `-W` | integer | Max seconds to honor a Retry-After header (default: unlimited). | None |
| `--no-wait` | boolean | Never wait for Retry-After headers; use fast exponential backoff instead. | `False` |
| `--license`, `-l` | text | Only download images whose license starts with this prefix (e.g. 'cc'); only applies to .jsonl input. | None |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst filter { #dtst-filter data-toc-label='filter' }

Filter images by moving rejects to a subfolder.

Evaluates images in a source folder against filter criteria and
moves those that fail into a subdirectory within the source
folder (default: filtered/). Filtered images can be restored
with --clear.

This is a non-destructive operation: no images are deleted, only
moved. The file explorer serves as the UI for reviewing what was
filtered. To undo individual decisions, move files back manually.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:
    dtst filter -d ./project --from faces --min-size 256
    dtst filter -d ./project --from faces --min-blur 50
    dtst filter -d ./project --from faces --min-size 256 --min-blur 50
    dtst filter -d ./project --from raw --max-tag microphone 0.25
    dtst filter -d ./project --from raw --max-tag illustration 0.2 --min-tag photograph 0.2
    dtst filter -d ./project --from faces --to rejects --min-size 256
    dtst filter config.yaml --min-size 128
    dtst filter -d ./project --from faces --clear
    dtst filter -d ./project --from faces --min-size 256 --dry-run

**Usage:**

```text
dtst filter [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--from` | text | Folder name to filter within the working directory. | None |
| `--to` | text | Subfolder name for rejected images. | None |
| `--min-size`, `-s` | integer | Minimum image dimension in pixels; images smaller are filtered out. | None |
| `--min-blur` | float | Minimum blur score (Laplacian variance) to keep; lower-scoring images are filtered as too blurry. | None |
| `--max-tag` | <text float> | Reject images where TAG score >= THRESHOLD (e.g. --max-tag microphone 0.25). | `()` |
| `--min-tag` | <text float> | Reject images where TAG score < THRESHOLD (e.g. --min-tag photograph 0.2). | `()` |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--clear` | boolean | Restore all filtered images back to the source folder. | `False` |
| `--dry-run` | boolean | Show what would be filtered without moving anything. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst frame { #dtst-frame data-toc-label='frame' }

Resize images to a target width and/or height.

Reads images from one or more source folders and writes resized
copies to a destination folder. Uses Lanczos resampling for
high-quality downscaling.

When both --width and --height are given, the --mode option controls
how aspect ratio differences are handled:

  stretch  Resize to exact dimensions, distorting if needed.
  crop     Scale to cover the target area, then trim excess (default).
  pad      Scale to fit within the target area, then fill the rest.

When only one dimension is given, the other is computed proportionally
and --mode is ignored.

Examples:

    dtst frame -d ./project --from faces --to resized -W 512 -H 512
    dtst frame -d ./project --from faces --to resized -W 512 -H 512 --mode pad --fill blur
    dtst frame -d ./project --from faces --to resized -W 512 -H 512 --mode crop --gravity top
    dtst frame -d ./project --from faces --to resized --width 512
    dtst frame config.yaml --dry-run

**Usage:**

```text
dtst frame [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--width`, `-W` | integer | Target width in pixels. If --height is omitted, aspect ratio is preserved. | None |
| `--height`, `-H` | integer | Target height in pixels. If --width is omitted, aspect ratio is preserved. | None |
| `--mode`, `-m` | choice (`stretch` &#x7C; `crop` &#x7C; `pad`) | Resize mode when both width and height are given (default: crop). | None |
| `--gravity`, `-g` | choice (`center` &#x7C; `top` &#x7C; `bottom` &#x7C; `left` &#x7C; `right` &#x7C; `top-left` &#x7C; `top-right` &#x7C; `bottom-left` &#x7C; `bottom-right`) | Anchor position for crop (part to keep) or pad (where to place image). Default: center. | None |
| `--fill`, `-f` | choice (`color` &#x7C; `edge` &#x7C; `reflect` &#x7C; `blur`) | Fill strategy for pad mode: color, edge, reflect, or blur (default: color). | None |
| `--fill-color` | text | Hex color for pad fill when --fill=color (default: #000000). | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--dry-run` | boolean | Preview what would be written without creating files. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst review { #dtst-review data-toc-label='review' }

Launch a web UI for manual image review.

Opens a local web server with an image grid. Click images to
select or deselect them, then apply to move filtered images
into a subfolder. Use the view toggle to switch between source
and filtered images to restore previously filtered images.

Press Ctrl+C to stop the server.

Examples:
    dtst review config.yaml
    dtst review -d ./project --from faces
    dtst review -d ./project --from faces --to rejected --port 9000
    dtst review config.yaml --no-open

**Usage:**

```text
dtst review [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Source folder name within working directory. | None |
| `--to` | text | Subfolder name for filtered images. | None |
| `--port`, `-p` | integer | Port for the web server. | None |
| `--no-open` | boolean | Do not open the browser automatically. | `False` |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst run { #dtst-run data-toc-label='run' }

Run a named workflow defined in a config file.

Executes a sequence of dtst commands and shell commands as defined
in the workflows section of the config file. Each command step
inherits its defaults from the corresponding config section unless
inherit: false is set.

Examples:
    dtst run pipeline config.yaml
    dtst run pipeline config.yaml --dry-run
    dtst run pipeline config.yaml -d ./my_dataset

**Usage:**

```text
dtst run [OPTIONS] WORKFLOW CONFIG
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Override working directory. | None |
| `--dry-run` | boolean | Print steps without executing. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst search { #dtst-search data-toc-label='search' }

Search for images across multiple engines.

Reads an optional YAML config file and generates image URLs from
Flickr, Serper (Google Images), Brave and Wikimedia Commons using
an expanded query matrix of search terms and suffixes.
Results are deduplicated and written to a JSONL file in the working
directory (default: results.jsonl) so multiple runs accumulate new
results.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Query matrix: By default, the command runs two kinds of queries for
each term: (1) the term alone, e.g. "chanterelle"; (2) the term
with each suffix, e.g. "chanterelle mushroom", "chanterelle forest".
Use --suffix-only to run only the second kind.

Examples:

    dtst search config.yaml
    dtst search config.yaml --dry-run
    dtst search config.yaml --max-pages 3 --engines flickr,wikimedia
    dtst search --terms "chanterelle" --suffixes "mushroom,forest" --engines brave -d ./chanterelle

**Usage:**

```text
dtst search [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--terms` | text | Comma-separated search terms (override config). | None |
| `--suffixes` | text | Comma-separated query suffixes (override config). | None |
| `--working-dir`, `-d` | path | Working directory where results are written (default: .). | None |
| `--output`, `-o` | text | Output filename within the working directory (default: results.jsonl). | None |
| `--max-pages`, `-m` | integer | Limit pages per engine per query. | None |
| `--engines`, `-e` | text | Comma-separated engine list (override config). | None |
| `--dry-run`, `-n` | boolean | Print query matrix and exit without searching. | `False` |
| `--workers`, `-w` | integer | Parallel workers (default: CPU count). | None |
| `--min-size`, `-s` | integer | Minimum image dimension in pixels (default: 512). | None |
| `--retries`, `-r` | integer | Number of retries per request (with exponential backoff). | `3` |
| `--timeout`, `-t` | float | Request timeout in seconds. | `30` |
| `--suffix-only` | boolean | Run only queries that include a suffix (e.g. 'term suffix'). Skip bare term queries. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst tag { #dtst-tag data-toc-label='tag' }

Score images against text labels using CLIP zero-shot classification.

Computes a similarity score for each image against each text label
and writes the results into per-image sidecar JSON files under a
"tags" key. Scores range from -1 to 1 (higher means stronger match).

Results are incremental — running with different label sets accumulates
scores in the sidecar. Use --force to recompute all labels.

Examples:
    dtst tag -d ./project --from raw --labels "microphone,photograph,illustration"
    dtst tag config.yaml
    dtst tag -d ./project --from raw --labels "cartoon,screenshot" --force
    dtst tag -d ./project --from raw --labels "microphone" --dry-run
    dtst tag -d ./project --from raw --clear

**Usage:**

```text
dtst tag [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Comma-separated source folders (supports globs, e.g. 'images/*'). | None |
| `--labels`, `-l` | text | Comma-separated text labels for zero-shot classification. | None |
| `--batch-size`, `-b` | integer | Images per inference batch. | None |
| `--force` | boolean | Recompute tags even if sidecar data already exists. | `False` |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--workers`, `-w` | integer | Number of threads for image preloading (default: 4). | None |
| `--clear` | boolean | Remove all tag data from sidecar files. | `False` |
| `--dry-run` | boolean | Preview what would be tagged without writing sidecars. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst upscale { #dtst-upscale data-toc-label='upscale' }

Upscale images using AI super-resolution models.

Reads images from one or more source folders and writes upscaled
copies to a destination folder. Uses spandrel to load PyTorch
super-resolution models (Real-ESRGAN, SwinIR, HAT, etc.).

By default uses a 4x Real-ESRGAN model. Use --scale to choose
between 2x and 4x upscaling, or --model to provide a custom
.pth weights file (scale is auto-detected from the model).

Use --denoise to control how much denoising is applied (4x only).
0.0 preserves the most texture, 1.0 applies full denoising.
This activates a lighter general-purpose model with controllable
denoise strength via weight interpolation.

Large images are processed in tiles to avoid GPU memory issues.
Adjust --tile-size to control memory usage (smaller = less VRAM).

Examples:
    dtst upscale -d ./project --from faces --to upscaled
    dtst upscale -d ./project --from faces --to upscaled --scale 2
    dtst upscale -d ./project --from faces --to upscaled --denoise 0.5
    dtst upscale -d ./project --from faces --to upscaled --model ./custom.pth
    dtst upscale config.yaml --dry-run

**Usage:**

```text
dtst upscale [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--scale`, `-s` | choice (`2` &#x7C; `4`) | Upscale factor. Ignored when --model is provided (default: 4). | None |
| `--model`, `-m` | text | Model preset name or path to a .pth file. Overrides --scale. | None |
| `--tile-size`, `-t` | integer | Tile size in pixels for processing; 0 disables tiling (default: 512). | None |
| `--tile-pad` | integer | Overlap padding between tiles in pixels (default: 32). | None |
| `--format`, `-f` | choice (`jpg` &#x7C; `png` &#x7C; `webp`) | Output image format. Default preserves the source format. | None |
| `--quality`, `-q` | integer | JPEG/WebP output quality, 1-100 (default: 95). | None |
| `--denoise`, `-n` | float | Denoise strength 0.0-1.0. Lower preserves more texture. Only available at 4x. | None |
| `--workers`, `-w` | integer | Number of threads for image preloading (default: 4). | None |
| `--dry-run` | boolean | Preview what would be written without processing. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

