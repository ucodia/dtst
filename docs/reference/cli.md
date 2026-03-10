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

When using --model clip, optional --prompt and --negative flags
accept text descriptions that shift the embedding space before
clustering. Positive prompts pull matching images closer together;
negative prompts push matching images apart. This helps merge
visually diverse images of the same concept into a single cluster.

Tuning clustering:
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
    dtst cluster -d ./bikes --model clip --prompt "motorcycle" --negative "car"
    dtst cluster config.yaml --model arcface --dry-run

**Usage:**

```text
dtst cluster [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory containing source folders and where output is written (default: .). | None |
| `--from` | text | Comma-separated source folder names within the working directory (default: faces). | None |
| `--to`, `-t` | text | Destination folder name within the working directory (default: clusters). | None |
| `--model`, `-m` | choice (`arcface` &#x7C; `clip`) | Embedding model for similarity (default: arcface). | None |
| `--top`, `-n` | integer | Maximum number of clusters to output; omit for all clusters. | None |
| `--min-cluster-size` | integer | Minimum images to form a cluster (default: 5). | None |
| `--min-samples` | integer | How many close neighbors a point needs to join a cluster; lower values include more borderline images (default: 2). | None |
| `--batch-size`, `-b` | integer | Images per inference batch (default: 32). | None |
| `--workers`, `-w` | integer | Number of workers for image preloading (default: CPU count). | None |
| `--prompt`, `-p` | text | Comma-separated positive text prompts to guide CLIP clustering toward (only with --model clip). | None |
| `--negative` | text | Comma-separated negative text prompts to guide CLIP clustering away from (only with --model clip). | None |
| `--dry-run` | boolean | Show image count and configuration without clustering. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst extract-faces { #dtst-extract-faces data-toc-label='extract-faces' }

Extract aligned face crops from images.

Detects faces in each image using MediaPipe (default) or dlib,
then produces an aligned and cropped face image for each detection.
The alignment normalises eye and mouth positions using the FFHQ
alignment technique.

Reads images from one or more source folders within the working
directory (default: raw) and writes face crops to a destination
folder (default: faces). Multiple source folders can be specified
as a comma-separated list with --from.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:
    dtst extract-faces config.yaml
    dtst extract-faces config.yaml --engine dlib --max-size 512
    dtst extract-faces -d ./crowd
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
| `--from` | text | Comma-separated source folder names within the working directory (default: raw). | None |
| `--to` | text | Destination folder name within the working directory (default: faces). | None |
| `--max-size`, `-M` | integer | Maximum side length in pixels; faces smaller than this are kept at natural size (default: no limit). | None |
| `--engine`, `-e` | choice (`mediapipe` &#x7C; `dlib`) | Face detection engine (default: mediapipe). | None |
| `--max-faces`, `-m` | integer | Max faces to extract per image (default: 1). | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--padding` / `--no-padding` | boolean | Enable/disable reflective padding on crops (default: enabled). | None |
| `--refine-landmarks` | boolean | Enable MediaPipe refined landmarks (478 vs 468). | `False` |
| `--debug` | boolean | Overlay landmark points on output images. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst fetch { #dtst-fetch data-toc-label='fetch' }

Download images from search results.

Reads results.jsonl from the working directory and downloads each
URL into a destination folder within that directory. Files are named
by the MD5 hash of the URL with the extension derived from the HTTP
Content-Type header. Existing files are skipped unless --force is set.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

When reading from results.jsonl, images can be filtered by known
dimensions (skipping images below the configured min_size) and by
license prefix (e.g. --license cc for Creative Commons only).

Per-domain throttling is applied automatically to respect server
rate limits (e.g. Wikimedia allows max 2 concurrent connections).
If a domain returns repeated 429 errors, remaining URLs for that
domain are skipped.

By default, Retry-After headers are honored. Use --max-wait to cap
the wait time, or --no-wait to skip waiting entirely. The two flags
are mutually exclusive.

Examples:

    dtst fetch config.yaml
    dtst fetch -d ./chanterelle
    dtst fetch -d ./chanterelle --to raw
    dtst fetch config.yaml --workers 16 --timeout 60
    dtst fetch config.yaml --force
    dtst fetch -d ./chanterelle --no-wait --license cc

**Usage:**

```text
dtst fetch [OPTIONS] [CONFIG]
```

**Options:**

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `--working-dir`, `-d` | path | Working directory where results.jsonl is read from and images are written to (default: .). | None |
| `--to` | text | Destination folder name within the working directory (default: raw). | None |
| `--min-size`, `-s` | integer | Minimum image dimension in pixels (default: 512). | None |
| `--workers`, `-w` | integer | Number of parallel download threads (default: CPU count). | None |
| `--timeout`, `-t` | integer | Per-request timeout in seconds. | `30` |
| `--force`, `-f` | boolean | Re-download files even if they already exist. | `False` |
| `--max-wait`, `-W` | integer | Max seconds to honor a Retry-After header (default: unlimited). | None |
| `--no-wait` | boolean | Never wait for Retry-After headers; use fast exponential backoff instead. | `False` |
| `--license`, `-l` | text | Only download images whose license starts with this prefix (e.g. 'cc'). | None |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst filter { #dtst-filter data-toc-label='filter' }

Filter images by moving rejects to a filtered/ subfolder.

Evaluates images in a source folder against filter criteria and
moves those that fail into a filtered/ subdirectory within the
source folder. Filtered images can be restored with --clear.

This is a non-destructive operation: no images are deleted, only
moved. The file explorer serves as the UI for reviewing what was
filtered. To undo individual decisions, move files back manually.

Can be invoked with just a config file, just CLI options, or both.
When both are provided, CLI options override config file values.

Examples:
    dtst filter -d ./project --from faces --min-size 256
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
| `--from` | text | Folder name to filter within the working directory (default: faces). | None |
| `--min-size`, `-s` | integer | Minimum image dimension in pixels; images smaller are filtered out. | None |
| `--workers`, `-w` | integer | Number of parallel workers (default: CPU count). | None |
| `--clear` | boolean | Restore all filtered images back to the source folder. | `False` |
| `--dry-run` | boolean | Show what would be filtered without moving anything. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

## dtst search { #dtst-search data-toc-label='search' }

Search for images across multiple engines.

Reads an optional YAML config file and generates image URLs from
Flickr, Serper (Google Images), Brave and Wikimedia Commons using
an expanded query matrix of search terms and suffixes.
Results are deduplicated and written to results.jsonl in the working
directory so multiple runs accumulate new results.

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
| `--working-dir`, `-d` | path | Working directory where results.jsonl is written (default: .). | None |
| `--max-pages`, `-m` | integer | Limit pages per engine per query. | None |
| `--engines`, `-e` | text | Comma-separated engine list (override config). | None |
| `--dry-run`, `-n` | boolean | Print query matrix and exit without searching. | `False` |
| `--workers`, `-w` | integer | Parallel workers (default: CPU count). | None |
| `--min-size`, `-s` | integer | Minimum image dimension in pixels (default: 512). | None |
| `--retries`, `-r` | integer | Number of retries per request (with exponential backoff). | `3` |
| `--timeout`, `-t` | float | Request timeout in seconds. | `30` |
| `--suffix-only` | boolean | Run only queries that include a suffix (e.g. 'term suffix'). Skip bare term queries. | `False` |
| `--help` | boolean | Show this message and exit. | `False` |

