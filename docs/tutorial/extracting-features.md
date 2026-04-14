# Extracting features

With your images collected, the next step is to extract meaningful structure from them. There are three paths depending on your use case:

- **Face datasets** — extract face crops, then cluster by identity using ArcFace
- **General image datasets** — cluster images directly by visual similarity using CLIP
- **Object/class datasets** — detect objects with `detect`, then extract class crops with `extract-classes`

Paths A and B produce numbered cluster folders sorted by size, ready for selection. Path C produces cropped images of specific object classes, which can then feed into clustering or selection.

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

## Path C: Object classes

When your dataset focuses on a specific type of object rather than faces or general scenes, you can use object detection to locate instances and crop them out.

### Detect objects

First, run `detect` to find objects in your images. This writes bounding boxes and confidence scores into each image's sidecar file:

```bash
dtst detect -d scratch/dahlias --from images --classes flower
```

After this, each sidecar JSON contains a `classes` key with detection results:

```json
{
  "source": "flickr",
  "classes": {
    "flower": [
      { "score": 0.92, "box": [120, 45, 580, 490] }
    ]
  }
}
```

You can detect multiple classes at once:

```bash
dtst detect -d scratch/dahlias --from images --classes flower,leaf,stem
```

### Extract class crops

The `extract-classes` command reads those detections and crops the corresponding regions from each image:

```bash
dtst extract-classes -d scratch/dahlias --from images --to flowers --classes flower
```

Each detected object becomes its own image in the output folder. The sidecar data is preserved with bounding box coordinates adjusted to match the cropped region.

### Square crops with margin

Two options help produce cleaner crops. Use `--square` to extend the shorter side of the bounding box to match the longer side, centering the object. Use `--margin` to add breathing room around the box as a ratio of the larger side:

```bash
# Square crops with 10% margin on each side
dtst extract-classes -d scratch/dahlias --from images --to flowers \
  --classes flower --square --margin 0.1
```

The margin is computed on the larger side of the box (after squaring if enabled) and applied equally on all four sides. For example, a 400×400 square box with `--margin 0.1` adds 40 pixels on each side, producing a 480×480 crop.

### Handling edge cases

When `--square` or `--margin` push the crop beyond the image boundary, the crop is clamped to the image edges by default. Use `--skip-partial` to discard those detections entirely:

```bash
dtst extract-classes -d scratch/dahlias --from images --to flowers \
  --classes flower --square --margin 0.1 --skip-partial
```

Filter out low-confidence detections with `--min-score`:

```bash
dtst extract-classes -d scratch/dahlias --from images --to flowers \
  --classes flower --min-score 0.5
```

Preview what would be extracted without writing any files:

```bash
dtst extract-classes -d scratch/dahlias --from images --to flowers \
  --classes flower --dry-run
```

### Cluster the crops

Once you have your class crops, you can cluster them by visual similarity using CLIP, the same way as Path B:

```bash
dtst cluster -d scratch/dahlias --from flowers --to cluster --model clip
```

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
    flowers/              <- class crops (Path C only)
    cluster/              <- grouped by similarity
      000/
      001/
      noise/
```
