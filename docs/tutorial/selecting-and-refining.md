# Selecting and refining

With your clusters ready, the next phase is selecting which groups to keep and refining their quality. This covers selecting clusters into a working folder, computing metadata, quality filtering, manual curation, and deduplication.

## Select clusters

Browse the cluster folders in your file explorer. Each numbered folder contains a group of similar images — pick the ones you want to keep.

Use the `select` command to consolidate chosen clusters into a `curated/` folder:

```bash
dtst select -d scratch/crowd --from cluster/000 --to curated
```

This copies images to the destination. Files that already exist by name in the destination are skipped. The original cluster output stays intact so you can experiment with different selections.

To move images instead of copying:

```bash
dtst select -d scratch/crowd --from cluster/000 --to curated --move
```

To preview what would happen:

```bash
dtst select -d scratch/crowd --from cluster/000 --to curated --dry-run
```

## Select with filters

You can combine selection with filter criteria to only select images that pass quality checks. This requires metadata from `analyze` and/or `detect` (see below).

Select only large, sharp images without microphones:

```bash
dtst select -d scratch/crowd --from faces --to curated \
  --min-size 1024 \
  --min-blur 5 \
  --max-detect microphone 0.25
```

This copies only images that are at least 1024px, have a blur score of 5 or above, and where the "microphone" detection score is below 0.25.

To preview what would be selected:

```bash
dtst select -d scratch/crowd --from faces --to curated --min-size 1024 --dry-run
```

## Analyze

The `analyze` command computes per-image metadata and stores it in JSON sidecar files alongside each image. This metadata is used by `select` (blur scores) and `dedup` (perceptual hashes).

```bash
dtst analyze -d scratch/crowd --from curated --phash --blur
```

After this runs, each image in `curated/` has a `.json` sidecar file containing its perceptual hash and blur score. Sidecars are merged incrementally — you can run `--phash` and `--blur` separately and both values accumulate.

To force recomputation on images that already have metadata:

```bash
dtst analyze -d scratch/crowd --from curated --phash --blur --force
```

## Detect

The `detect` command finds specific objects in images using OWL-ViT 2 open-vocabulary object detection. It localizes objects and returns a confidence score and bounding box.

```bash
dtst detect -d scratch/crowd --from curated --classes "microphone,chair"
```

Results are written to sidecar files under a `classes` key. Each class gets a list of detections (score + bounding box) sorted by confidence, or an empty list if not found. You can filter with `select`:

```bash
dtst select -d scratch/crowd --from curated --to clean --max-detect microphone 0.5
```

This excludes images where a microphone was detected with confidence >= 0.5.

## Review

The `review` command launches an interactive web UI for manual review of borderline images that automated filters can't catch.

```bash
dtst review -d scratch/crowd --from curated
```

This opens a local web server with an image grid. Click images to select or deselect them, then apply to move filtered images into a subfolder. Use the view toggle to switch between source and filtered images to restore previously filtered ones.

Press Ctrl+C to stop the server.

## Dedup

The `dedup` command finds near-duplicate images using perceptual hash similarity and keeps only the best copy from each group. This requires phash data from the `analyze` step.

```bash
dtst dedup -d scratch/crowd --from curated
```

For each group of duplicates, the winner is chosen by resolution first, then file size, then blur sharpness (if available). Losers are moved to `curated/duplicated/` by default.

To use a stricter threshold (lower value = images must be more similar to count as duplicates):

```bash
dtst dedup -d scratch/crowd --from curated --threshold 4
```

To preview or undo:

```bash
dtst dedup -d scratch/crowd --from curated --dry-run
dtst dedup -d scratch/crowd --from curated --clear
```

## Directory so far

```
scratch/
  crowd/
    results.jsonl
    images/
      search1/
      search2/
      frames/
    videos/
    faces/
    cluster/
      000/
      001/
      noise/
    curated/              <- selected, reviewed and deduplicated
      *.json              <- sidecar metadata (phash, blur, detections)
      rejected/           <- images removed by review
      duplicated/         <- images removed by dedup
```
