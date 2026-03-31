# Final preparation

The last step is to expand the dataset with augmentations and produce resized versions at the dimensions you need for training.

## Augment

The `augment` command increases dataset size by applying image transformations. It reads from one or more source folders and writes the transformed images (plus the originals by default) to a destination folder.

To create the final 1024px dataset with horizontal flips (2x the images):

```bash
dtst augment -d scratch/crowd --from select --to final/1024 --flipX
```

You can combine multiple transforms in a single run. This produces the original plus three variants of each image (4x the dataset):

```bash
dtst augment -d scratch/crowd --from select --to final/1024 --flipX --flipY --flipXY
```

If you only want the transformed images without copying the originals:

```bash
dtst augment -d scratch/crowd --from select --to final/1024 --flipX --no-copy
```

## Resize

The `frame` command resizes images to a target width and/or height using Lanczos resampling. Use it to produce smaller versions of the augmented dataset:

```bash
dtst frame -d scratch/crowd --from final/1024 --to final/512 --width 512 --height 512
dtst frame -d scratch/crowd --from final/1024 --to final/256 --width 256 --height 256
```

When only one dimension is given, the other is computed proportionally to preserve the aspect ratio:

```bash
dtst frame -d scratch/crowd --from final/1024 --to final/512 --width 512
```

## Final directory

```
crowd.yaml
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
    select/
    final/
      1024/              <- augmented originals
      512/               <- resized to 512px
      256/               <- resized to 256px
```

