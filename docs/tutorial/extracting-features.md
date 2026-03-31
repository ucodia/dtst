# Extracting features

With your images collected, the next step is to extract meaningful structure from them. There are two paths depending on your use case:

- **Face datasets** — extract face crops, then cluster by identity using ArcFace
- **General image datasets** — cluster images directly by visual similarity using CLIP

Both paths produce numbered cluster folders sorted by size, ready for selection.

## Path A: Faces

### Extract faces

The `extract-faces` command detects faces in each image and produces aligned and cropped face images. Use the `--from` glob pattern `images/*` to capture all image subfolders in a single pass:

```bash
dtst extract-faces -d scratch/crowd --from images/* --to faces
```

After this runs, `scratch/crowd/faces/` contains one cropped face image per detection. Images with no detected faces are skipped.

A few options that are useful in practice:

```bash
# Limit to one face per image
dtst extract-faces -d scratch/crowd --from images/* --to faces --max-faces 1

# Use dlib instead of MediaPipe for detection
dtst extract-faces -d scratch/crowd --from images/* --to faces --engine dlib

# Skip faces whose crop extends beyond the image boundary
dtst extract-faces -d scratch/crowd --from images/* --to faces --skip-partial
```

The default engine is MediaPipe, which is faster. Dlib can be more accurate on challenging angles. You can try both by writing to different output folders:

```bash
dtst extract-faces -d scratch/crowd --from images/* --to faces-dlib --engine dlib
```

### Cluster by identity

Use the default `arcface` model to group faces by identity — photos of the same person end up in the same cluster:

```bash
dtst cluster -d scratch/crowd --from faces --to cluster
```

## Path B: General images

### Cluster by visual similarity

For objects, scenes, styles, or any non-face dataset, use the `clip` model to cluster images directly:

```bash
dtst cluster -d scratch/crowd --from images/* --to cluster --model clip
```

No face extraction step is needed — CLIP works on any image content.

## Cluster output

Whichever path you chose, `scratch/crowd/cluster/` now contains numbered folders sorted by size (000 is the largest). Images that don't fit any cluster are placed in `noise/`.

### Tuning the clusters

```bash
# Only keep the top 5 largest clusters
dtst cluster -d scratch/crowd --from faces --to cluster --top 5

# Require larger clusters (minimum 20 images to form a cluster)
dtst cluster -d scratch/crowd --from faces --to cluster --min-cluster-size 20
```

Use `--clean` to remove the output directory before writing, which is useful when re-running with different parameters:

```bash
dtst cluster -d scratch/crowd --from faces --to cluster --min-cluster-size 20 --clean
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
    faces/                <- face crops (Path A only)
    cluster/              <- grouped by similarity
      000/
      001/
      noise/
```
