# Get started

`dtst` is a Python toolkit for researching, collecting, curating, and preparing image datasets for machine learning applications such as StyleGAN training and computer vision research.

This toolkit provides a complete set of independent tools for dataset preparation, from initial data collection to final preprocessing. Each tool can be used standalone or combined in custom workflows based on your specific needs

This guide walks you through building your first dataset. By the end you will have searched for images, downloaded them, and extracted aligned face crops, all within a single working directory.

## Prerequisites

Install `dtst` and copy the `.env.example` file to `.env`, then fill in the API keys for whichever search engines you want to use:

```bash
cp .env.example .env
```

## Create a working directory

Every `dtst` pipeline lives in a working directory. This is a plain folder on disk and also your database. You can open it in the file explorer, drop files in manually, and the tools will pick them up.

```bash
mkdir crowd
```

All the commands below use `-d crowd` to point at this folder.

## Step 1: Search

The `search` command queries one or more image search engines and writes the results to `crowd/results.jsonl`. Nothing is downloaded yet; this step just collects URLs.

```bash
dtst search -d crowd \
  --terms "crowd,street crowd,urban crowd" \
  --suffixes "photography,candid" \
  --engines flickr,brave
```

You can re-run this command as many times as you like. Results accumulate and are deduplicated automatically, so adding more engines or terms later just extends the file without creating duplicates.

To preview the query matrix without running any searches:

```bash
dtst search -d crowd \
  --terms "crowd,street crowd" \
  --suffixes "photography,candid" \
  --engines flickr \
  --dry-run
```

## Step 2: Fetch

Once you have a `results.jsonl`, the `fetch` command downloads each image into a folder inside your working directory.

```bash
dtst fetch -d crowd --to raw
```

After this runs, `crowd/raw/` contains all the successfully downloaded images. Files are named by a hash of their source URL so re-running fetch is safe; already-downloaded images are skipped automatically.

You can also filter downloads by minimum image size or license:

```bash
dtst fetch -d crowd --to raw --min-size 1024 --license cc
```

### Adding images manually

Because the working directory is the source of truth, you can supplement the fetched images by simply copying files into any folder. For example, drop additional crowd photos into `crowd/extra/`. No registration or import command is needed.

## Step 3: Extract faces

The `extract-faces` command reads images from one or more source folders and writes aligned face crops to an output folder.

```bash
dtst extract-faces -d crowd --from raw --to faces
```

To include your manually added images alongside the fetched ones, pass both folders with `--from`:

```bash
dtst extract-faces -d crowd --from raw,extra --to faces
```

After this runs, `crowd/faces/` contains one cropped and aligned face image per detection. Images with no detected faces are skipped.

### Tuning the extraction

A few options that are useful in practice:

```bash
# Limit to one face per image, use a larger output size
dtst extract-faces -d crowd --from raw --to faces --max-faces 1 --max-size 512

# Use dlib instead of MediaPipe for detection
dtst extract-faces -d crowd --from raw --to faces --engine dlib
```

## Step 4: Analyze

The `analyze` command computes per-image metadata and stores it in JSON sidecar files alongside each image. This metadata is used by later steps such as `filter` (blur scores) and `dedup` (perceptual hashes).

To compute both perceptual hashes and blur scores for all face crops:

```bash
dtst analyze -d crowd --from faces --phash --blur
```

After this runs, each image in `crowd/faces/` has a `.json` sidecar file containing its phash and blur score. Sidecars are merged incrementally, so you can run `--phash` and `--blur` separately and both values accumulate.

Images that already have the requested metadata are skipped automatically. To force recomputation:

```bash
dtst analyze -d crowd --from faces --phash --blur --force
```

## Step 5: Filter

The `filter` command lets you remove images that don't meet certain criteria. Rather than deleting them, it moves rejects to a subfolder within the source folder. This keeps everything non-destructive and easy to undo.

For example, to remove face crops smaller than 1024 pixels:

```bash
dtst filter -d crowd --from faces --to filtered --min-size 1024
```

You can also filter by blur score to remove blurry images. This requires blur data from the `analyze` step:

```bash
dtst filter -d crowd --from faces --to filtered --min-blur 50
```

Both criteria can be combined in a single run:

```bash
dtst filter -d crowd --from faces --to filtered --min-size 1024 --min-blur 50
```

After this runs, images below the threshold are in `crowd/faces/filtered/`. You can review them in the file explorer and move any back if the filter was too aggressive.

To preview what would be filtered without moving anything:

```bash
dtst filter -d crowd --from faces --to filtered --min-size 1024 --dry-run
```

To undo filtering and restore all images back to the source folder:

```bash
dtst filter -d crowd --from faces --to filtered --clear
```

## Step 6: Dedup

The `dedup` command finds near-duplicate images using perceptual hash similarity and keeps only the best copy from each group. This requires phash data from the `analyze` step.

```bash
dtst dedup -d crowd --from faces --to duplicated
```

For each group of duplicates, the winner is chosen by resolution first, then file size, then blur sharpness (if blur scores are available from `analyze`). Losers are moved to the `--to` subfolder within the source folder.

To use a stricter threshold (lower value means images must be more similar to be considered duplicates):

```bash
dtst dedup -d crowd --from faces --to duplicated --threshold 4
```

To preview what would be deduplicated without moving anything:

```bash
dtst dedup -d crowd --from faces --to duplicated --dry-run
```

To undo deduplication and restore all images:

```bash
dtst dedup -d crowd --from faces --to duplicated --clear
```

## Step 7: Cluster

The `cluster` command groups similar images together for easier curation. It computes embeddings for each image, runs unsupervised clustering, and writes each cluster to a numbered subdirectory sorted by size (000 is the largest).

For face datasets, the default `arcface` model clusters by identity, grouping photos of the same person together:

```bash
dtst cluster -d crowd --from faces --to clusters
```

After this runs, `crowd/clusters/` contains folders like `000/`, `001/`, etc., each with images of a distinct person. Images that don't fit any cluster are placed in `noise/`. You can browse the folders, keep the ones you want, and delete the rest.

To limit the output to the top 3 clusters:

```bash
dtst cluster -d crowd --from faces --to clusters --top 3
```

For general image datasets (not faces), use the `clip` model which clusters by visual similarity:

```bash
dtst cluster -d crowd --from raw --to clusters --model clip
```

## The resulting layout

After running all seven steps your working directory looks like this:

```
crowd/
  results.jsonl       <- search output, accumulates across runs
  raw/                <- images downloaded by fetch
  extra/              <- images you added manually
  faces/              <- aligned face crops from extract-faces
    *.json            <- sidecar metadata from analyze (phash, blur)
    filtered/         <- images removed by filter
    duplicated/       <- images removed by dedup
  clusters/           <- grouped by similarity from cluster
    000/              <- largest cluster
    001/              <- second largest
    noise/            <- unclustered images
    clusters.json     <- cluster metadata
```

Each folder is self-contained and can be inspected, filtered, or extended at any time. If you want to experiment with different extraction settings, point `--to` at a new folder to keep both versions side by side:

```bash
dtst extract-faces -d crowd --from raw,extra --to faces-dlib --engine dlib
```

## Using a config file

For repeatable pipelines, collect your parameters into a YAML config file:

```yaml
# crowd.yaml
working_dir: "./crowd"

search:
  terms:
    - crowd
    - street crowd
    - urban crowd
  suffixes:
    - photography
    - candid
  engines:
    - flickr
    - brave
  min_size: 512

fetch:
  to: raw
  min_size: 1024

extract_faces:
  from:
    - raw
    - extra
  to: faces
  engine: mediapipe
  max_faces: 1
  max_size: 512

analyze:
  from: faces
  phash: true
  blur: true

filter:
  from: faces
  to: filtered
  min_size: 1024
  min_blur: 50

dedup:
  from: faces
  to: duplicated
  threshold: 8

cluster:
  from: faces
  to: clusters
  model: arcface
  min_cluster_size: 16
```

Then run each stage by passing the config file:

```bash
dtst search crowd.yaml
dtst fetch crowd.yaml
dtst extract-faces crowd.yaml
dtst analyze crowd.yaml
dtst filter crowd.yaml
dtst dedup crowd.yaml
dtst cluster crowd.yaml
```

CLI options can still be used alongside the config file and will override the corresponding config values.
