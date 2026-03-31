# Selecting and refining

With your clusters ready, the next phase is selecting which groups to keep and refining their quality. This covers copying clusters into a working folder, computing metadata, tagging, filtering, manual curation, and deduplication. All refinement commands are non-destructive — rejects are moved to subfolders and can be restored.

## Copy selected clusters

Browse the cluster folders in your file explorer. Each numbered folder contains a group of similar images — pick the ones you want to keep.

Use the `copy` command to consolidate chosen clusters into a `select/` folder:

```bash
dtst copy -d scratch/crowd --from cluster/000 --to select
```

This copies images without any transformation. Files that already exist by name in the destination are skipped. The original cluster output stays intact so you can experiment with different selections.

To preview what would be copied:

```bash
dtst copy -d scratch/crowd --from cluster/000 --to select --dry-run
```

## Analyze

The `analyze` command computes per-image metadata and stores it in JSON sidecar files alongside each image. This metadata is used by `filter` (blur scores) and `dedup` (perceptual hashes).

```bash
dtst analyze -d scratch/crowd --from select --phash --blur
```

After this runs, each image in `select/` has a `.json` sidecar file containing its perceptual hash and blur score. Sidecars are merged incrementally — you can run `--phash` and `--blur` separately and both values accumulate.

To force recomputation on images that already have metadata:

```bash
dtst analyze -d scratch/crowd --from select --phash --blur --force
```

## Tag

The `tag` command scores each image against text labels using CLIP zero-shot classification. This is useful for detecting unwanted content (cartoons, microphones, screenshots) that you can then filter out.

```bash
dtst tag -d scratch/crowd --from select --labels "cartoon,microphone"
```

Scores are written to the same sidecar JSON files under a `tags` key. Scores range from -1 to 1 (higher means stronger match). Running with different label sets accumulates scores.

## Filter

The `filter` command evaluates images against criteria and moves failures to a subfolder. This keeps everything non-destructive and easy to undo.

Filter by size, blur, and tag scores:

```bash
dtst filter -d scratch/crowd --from select \
  --min-size 1024 \
  --min-blur 5 \
  --max-tag microphone 0.25
```

This removes images smaller than 1024px, images with a blur score below 5, and images where the "microphone" tag score is 0.25 or higher. Rejects are moved to `select/filtered/` by default.

You can combine `--max-tag` (reject if score is too high) with `--min-tag` (reject if score is too low):

```bash
dtst filter -d scratch/crowd --from select \
  --max-tag cartoon 0.2 \
  --min-tag photograph 0.3
```

To preview what would be filtered without moving anything:

```bash
dtst filter -d scratch/crowd --from select --min-size 1024 --dry-run
```

To undo filtering and restore all images:

```bash
dtst filter -d scratch/crowd --from select --clear
```

## Review

The `review` command launches an interactive web UI for manual review of borderline images that automated filters can't catch.

```bash
dtst review -d scratch/crowd --from select
```

This opens a local web server with an image grid. Click images to select or deselect them, then apply to move filtered images into a subfolder. Use the view toggle to switch between source and filtered images to restore previously filtered ones.

Press Ctrl+C to stop the server.

## Dedup

The `dedup` command finds near-duplicate images using perceptual hash similarity and keeps only the best copy from each group. This requires phash data from the `analyze` step.

```bash
dtst dedup -d scratch/crowd --from select
```

For each group of duplicates, the winner is chosen by resolution first, then file size, then blur sharpness (if available). Losers are moved to `select/duplicated/` by default.

To use a stricter threshold (lower value = images must be more similar to count as duplicates):

```bash
dtst dedup -d scratch/crowd --from select --threshold 4
```

To preview or undo:

```bash
dtst dedup -d scratch/crowd --from select --dry-run
dtst dedup -d scratch/crowd --from select --clear
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
    select/               <- reviewed and deduplicated
      *.json              <- sidecar metadata (phash, blur, tags)
      filtered/           <- images removed by filter
      filtered_manual/    <- images removed by review
      duplicated/         <- images removed by dedup
```
