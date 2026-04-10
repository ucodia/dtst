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

Compute image metrics and write JSON sidecars.

Analyzes images in the source folders and writes per-image sidecar
JSON files containing the requested metrics. Sidecars are merged
incrementally — running with different metrics accumulates results.

CPU metrics: phash, blur.
IQA metrics (GPU-accelerated): any metric from IQA-PyTorch (e.g.
musiq, clipiqa, topiq_nr, dbcnn, hyperiqa, niqe, brisque).

Examples:
  dtst analyze --from raw --metrics phash,blur -d ./my-dataset
  dtst analyze config.yaml --metrics phash
  dtst analyze --from raw --metrics musiq,clipiqa -d ./my-dataset
  dtst analyze --from raw --metrics phash,blur,musiq --force
  dtst analyze --from raw --clear -d ./my-dataset

**Usage:**

```text
dtst analyze [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Comma-separated source folders (supports globs, e.g. 'images/*'). | None |
| `--metrics`, `-m` | text | Comma-separated metric names (e.g. 'phash,blur,musiq,clipiqa'). | None |
| `--force` | boolean | Recompute all metrics even if sidecar data already exists. | `False` |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--workers`, `-w` | integer | Number of parallel workers for CPU metrics (default: CPU count). | None |
| `--clear` | boolean | Remove all sidecar files from source folders. | `False` |
| `--dry-run` | boolean | Preview what would be computed without writing sidecars. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst annotate { #dtst-annotate data-toc-label='annotate' }

Write source and license metadata into image sidecars.

Annotates all images in the given folders with provenance metadata
(source, license, origin). Useful for manually imported images that
were not fetched through the pipeline. Sidecars are merged
incrementally — existing fields are preserved unless --overwrite
is used.

At least one of --source, --license, or --origin is required.

Examples:
    dtst annotate --from extra --source "unsplash" --license "cc0" -d ./my-dataset
    dtst annotate config.yaml
    dtst annotate --from raw,extra --source "personal" --license "all-rights-reserved"
    dtst annotate --from extra --source "flickr" --overwrite -d ./my-dataset
    dtst annotate --from extra --source "personal" --dry-run

**Usage:**

```text
dtst annotate [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Comma-separated source folders (supports globs, e.g. 'images/*'). | None |
| `--source`, `-s` | text | Source name to write (e.g. 'unsplash', 'personal'). | None |
| `--license`, `-l` | text | License string to write (e.g. 'cc-by', 'cc0', 'all-rights-reserved'). | None |
| `--origin`, `-o` | text | Origin URL to write (applied to all images). | None |
| `--overwrite` | boolean | Overwrite existing source/license/origin values in sidecars. | `False` |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--dry-run` | boolean | Preview what would be annotated without writing sidecars. | `False` |
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

## dtst dedup { #dtst-dedup data-toc-label='dedup' }

Deduplicate images by perceptual hash similarity.

Groups images by phash hamming distance and keeps the best image
from each duplicate group. By default, original (non-upscaled)
images are preferred; use --prefer-upscaled to reverse this. Within
each preference tier, the winner is chosen by resolution
(width x height), then file size, then blur sharpness. Losers are
moved to a duplicated/ subdirectory within the source folder
(configurable with --to).

Requires phash sidecar data from ``dtst analyze --metrics phash``. Blur
scores (from ``dtst analyze --metrics blur``) are used as a tiebreaker
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
| `--prefer-upscaled` | boolean | Prefer upscaled images over originals when deduplicating. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst detect { #dtst-detect data-toc-label='detect' }

Detect objects in images using OWL-ViT 2.

Uses open-vocabulary object detection to find specific objects in images
and writes the results into per-image sidecar JSON files under a
"classes" key. Each class gets all detections (score + bounding box)
sorted by confidence, or null if not found.

Each run replaces the entire "classes" key in the sidecar.

Examples:
    dtst detect -d ./project --from raw --classes "microphone,chair,table"
    dtst detect config.yaml
    dtst detect -d ./project --from raw --classes "microphone" --threshold 0.4
    dtst detect -d ./project --from raw --classes "microphone" --dry-run
    dtst detect -d ./project --from raw --clear

**Usage:**

```text
dtst detect [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Comma-separated source folders (supports globs, e.g. 'images/*'). | None |
| `--classes`, `-c` | text | Comma-separated object classes to detect (e.g. 'microphone,chair'). | None |
| `--threshold` | float | Minimum detection confidence. | None |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--workers`, `-w` | integer | Number of threads for image preloading (default: 4). | None |
| `--max-instances` | integer | Maximum detections per class per image. | None |
| `--clear` | boolean | Remove all detection data from sidecar files. | `False` |
| `--dry-run` | boolean | Preview what would be detected without writing sidecars. | `False` |
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
    dtst fetch -d ./chanterelle --to raw --input results.jsonl
    dtst fetch -d ./project --to videos --input urls.txt
    dtst fetch config.yaml --workers 16 --timeout 60
    dtst fetch config.yaml --force
    dtst fetch -d ./chanterelle --to raw --input results.jsonl --no-wait --license cc

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

## dtst format { #dtst-format data-toc-label='format' }

Convert and normalize image formats, channels, and metadata.

Reads images from source folders and writes converted copies to a
destination folder.  Can change format (jpg/png/webp), enforce
channel mode (rgb/grayscale), and strip EXIF metadata.

When --format is omitted the source format is preserved, but other
transformations (--channels, --strip-metadata) still apply.

Examples:
    dtst format -d ./project --from raw --to formatted -f jpg -q 90
    dtst format -d ./project --from raw --to clean --strip-metadata --channels rgb
    dtst format -d ./project --from raw --to gray --channels grayscale
    dtst format config.yaml --dry-run

**Usage:**

```text
dtst format [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--format`, `-f` | choice (`jpg` &#x7C; `png` &#x7C; `webp`) | Output image format. When omitted the source format is preserved. | None |
| `--quality`, `-q` | integer | JPEG/WebP output quality, 1-100 (default: 95). Ignored for PNG. | None |
| `--compress-level` | integer | PNG compression level, 0 (none) to 9 (max). Default: 0. Ignored for JPEG/WebP. | None |
| `--strip-metadata` | boolean | Remove EXIF data and embedded ICC profiles from output images. | `False` |
| `--channels`, `-c` | choice (`rgb` &#x7C; `grayscale`) | Enforce channel mode. 'rgb' converts to 3-channel RGB (drops alpha). 'grayscale' converts to single-channel. | None |
| `--background` | text | Background color for alpha compositing (default: white). Accepts named colors or hex codes. | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--dry-run` | boolean | Preview what would be written without creating files. | `False` |
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
| `--gravity`, `-g` | choice (`center` &#x7C; `top` &#x7C; `bottom` &#x7C; `left` &#x7C; `right`) | Anchor position for crop (part to keep) or pad (where to place image). Default: center. | None |
| `--fill`, `-f` | choice (`color` &#x7C; `edge` &#x7C; `reflect` &#x7C; `blur`) | Fill strategy for pad mode: color, edge, reflect, or blur (default: color). | None |
| `--fill-color` | text | Hex color for pad fill when --fill=color (default: #000000). | None |
| `--quality`, `-q` | integer | JPEG/WebP output quality, 1-100 (default: 95). Ignored for PNG. | None |
| `--compress-level` | integer | PNG compression level, 0 (none) to 9 (max). Default: 0. Ignored for JPEG/WebP. | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--dry-run` | boolean | Preview what would be written without creating files. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst rename { #dtst-rename data-toc-label='rename' }

Sequentially rename images in-place with a prefix and zero-padded number.

Renames all images in the given folders to {prefix}{number}.{ext},
where the number is zero-padded to the specified number of digits.
Sidecar JSON files are moved along with their images. Operates
in-place — there is no --to option.

Examples:
    dtst rename --from raw --prefix "img_" -d ./my-dataset
    dtst rename --from raw --prefix "photo_" --digits 5 -d ./my-dataset
    dtst rename config.yaml --dry-run
    dtst rename --from faces --prefix "face_" -n 4

**Usage:**

```text
dtst rename [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Comma-separated source folders (supports globs, e.g. 'images/*'). | None |
| `--prefix`, `-p` | text | Filename prefix for renamed files (default: ''). | None |
| `--digits`, `-n` | integer | Number of zero-padded digits (default: auto based on total count). | None |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--dry-run` | boolean | Preview renames without executing. | `False` |
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
| `--to` | text | Subfolder name for filtered images. | `rejected` |
| `--port`, `-p` | integer | Port for the web server. | `8888` |
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
Flickr, Serper (Google Images), Brave, Wikimedia Commons, and
iNaturalist. Text-based engines use an expanded query matrix of
search terms and suffixes. iNaturalist uses taxon IDs instead.
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
    dtst search --taxon-ids 47169,54743 -d ./fungi

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
| `--taxon-ids` | text | Comma-separated iNaturalist taxon IDs (implies --engines inaturalist). | None |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst select { #dtst-select data-toc-label='select' }

Select images from source folders into a destination folder.

Copies (or moves with --move) images from one or more source folders
into a destination folder. When filter criteria are provided, only
images that pass all criteria are selected. Without criteria, all
images are selected.

Files that already exist in the destination (by name) are skipped.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:
    dtst select -d ./project --from raw --to backup
    dtst select -d ./project --from raw,extra --to combined
    dtst select -d ./project --from faces --to curated --min-side 256
    dtst select -d ./project --from faces --to curated --min-metric blur 5
    dtst select -d ./project --from faces --to curated --min-metric blur 5 --min-metric musiq 60
    dtst select -d ./project --from faces --to curated --max-metric brisque 40
    dtst select -d ./project --from raw --to clean --max-detect microphone 0.5
    dtst select -d ./project --from raw --to licensed --source serper,flickr
    dtst select config.yaml --dry-run

**Usage:**

```text
dtst select [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folders within the working directory (supports globs, e.g. 'images/*'). | None |
| `--to` | text | Destination folder name within the working directory. | None |
| `--move` | boolean | Move images instead of copying (removes originals). | `False` |
| `--min-side`, `-s` | integer | Minimum largest side in pixels; images with max(w,h) below this are excluded. | None |
| `--max-side` | integer | Maximum largest side in pixels; images with max(w,h) above this are excluded. | None |
| `--min-width` | integer | Minimum width in pixels; narrower images are excluded. | None |
| `--max-width` | integer | Maximum width in pixels; wider images are excluded. | None |
| `--min-height` | integer | Minimum height in pixels; shorter images are excluded. | None |
| `--max-height` | integer | Maximum height in pixels; taller images are excluded. | None |
| `--min-metric` | <text float> | Minimum metric threshold (e.g. --min-metric blur 5). Can be repeated. | `()` |
| `--max-metric` | <text float> | Maximum metric threshold (e.g. --max-metric brisque 40). Can be repeated. | `()` |
| `--max-detect` | <text float> | Exclude images where detection score >= THRESHOLD (e.g. --max-detect microphone 0.5). | `()` |
| `--min-detect` | <text float> | Exclude images where detection score < THRESHOLD (e.g. --min-detect chair 0.3). | `()` |
| `--source` | text | Comma-separated list of sources to include (e.g. 'serper,flickr'); checked against sidecar 'source' field. | None |
| `--license` | text | Comma-separated list of licenses to include (e.g. 'cc-by,none'); checked against sidecar 'license' field. | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--dry-run` | boolean | Preview what would be selected without creating files. | `False` |
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

## dtst validate { #dtst-validate data-toc-label='validate' }

Validate that all images in a folder are consistent.

Checks that every image shares the same dimensions and channel mode.
Optionally checks that images are square. Warns if any PNG files use
compression (which slows down loading).

Examples:
    dtst validate --from faces -d ./my-dataset
    dtst validate --from faces --square -d ./my-dataset
    dtst validate config.yaml

**Usage:**

```text
dtst validate [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--from` | text | Comma-separated source folders (supports globs, e.g. 'images/*'). | None |
| `--working-dir`, `-d` | path | Working directory (default: .). | None |
| `--square` | boolean | Check that all images are square (width == height). | `False` |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--help` | boolean | Show this message and exit. | `False` |

