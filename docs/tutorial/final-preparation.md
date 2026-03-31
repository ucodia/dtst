# Final preparation

The last step is to expand the dataset with augmentations, optionally upscale images, and produce resized versions at the dimensions you need for training.

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

## Upscale

The `upscale` command increases image resolution using AI super-resolution models. This is useful when source images are too small for your target training resolution — for example, face crops that came out at 256px but you need 1024px.

To upscale images 4x (the default):

```bash
dtst upscale -d scratch/crowd --from final/1024 --to final/upscaled
```

For 2x upscaling:

```bash
dtst upscale -d scratch/crowd --from final/1024 --to final/upscaled --scale 2
```

The default 4x model (Real-ESRGAN) tends to smooth out textures, especially on noisy source images. Use `--denoise` to control how much denoising is applied. Lower values preserve more natural texture, which is particularly important for face datasets:

```bash
dtst upscale -d scratch/crowd --from final/1024 --to final/upscaled --denoise 0
```

The `--denoise` option accepts a value between 0.0 and 1.0:

- `0.0` — maximum texture preservation (recommended for faces)
- `0.5` — balanced
- `1.0` — full denoising (smoothest result)

Note that `--denoise` is only available with 4x upscaling and activates a different, lighter model (realesr-general-x4v3). It cannot be combined with `--model` or `--scale 2`.

Large images are processed in tiles to avoid GPU memory issues. If you run into out-of-memory errors, reduce the tile size:

```bash
dtst upscale -d scratch/crowd --from final/1024 --to final/upscaled --tile-size 256
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
      upscaled/          <- AI-upscaled (optional)
      512/               <- resized to 512px
      256/               <- resized to 256px
```

