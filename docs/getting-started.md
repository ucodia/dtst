# Getting started

This guide walks you through building your first image dataset with `dtst`. By the end you will have searched for images, downloaded them, and extracted aligned face crops, all within a single working directory.

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

Once you have a `results.jsonl`, the `fetch` command downloads each image into a folder inside your working directory. By default that folder is `raw/`.

```bash
dtst fetch -d crowd
```

After this runs, `crowd/raw/` contains all the successfully downloaded images. Files are named by a hash of their source URL so re-running fetch is safe; already-downloaded images are skipped automatically.

You can also filter downloads by minimum image size or license:

```bash
dtst fetch -d crowd --min-size 1024 --license cc
```

### Adding images manually

Because the working directory is the source of truth, you can supplement the fetched images by simply copying files into any folder. For example, drop additional crowd photos into `crowd/extra/`. No registration or import command is needed.

## Step 3: Extract faces

The `extract-faces` command reads images from one or more source folders and writes aligned face crops to an output folder. It defaults to reading from `raw/` and writing to `faces/`.

```bash
dtst extract-faces -d crowd
```

To include your manually added images alongside the fetched ones, pass both folders with `--from`:

```bash
dtst extract-faces -d crowd --from raw,extra
```

After this runs, `crowd/faces/` contains one cropped and aligned face image per detection. Images with no detected faces are skipped.

### Tuning the extraction

A few options that are useful in practice:

```bash
# Limit to one face per image, use a larger output size
dtst extract-faces -d crowd --max-faces 1 --max-size 512

# Use dlib instead of MediaPipe for detection
dtst extract-faces -d crowd --engine dlib
```

## The resulting layout

After running all three steps your working directory looks like this:

```
crowd/
  results.jsonl       ← search output, accumulates across runs
  raw/                ← images downloaded by fetch
  extra/              ← images you added manually
  faces/              ← aligned face crops from extract-faces
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
```

Then run each stage by passing the config file:

```bash
dtst search crowd.yaml
dtst fetch crowd.yaml
dtst extract-faces crowd.yaml
```

CLI options can still be used alongside the config file and will override the corresponding config values.
