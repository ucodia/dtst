# Final preparation

The last steps are to expand the dataset with augmentations, optionally upscale images, rename files with clean sequential names, normalize image formats and channels, produce resized versions at the dimensions you need for training, and validate the result.

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

## Rename

The `rename` command gives images clean, sequential filenames with a consistent prefix. Renaming early — before format and resize — means every downstream folder inherits the clean names automatically. It operates in-place — there is no `--to` option.

To rename all images in `final/1024` with a "crowd_" prefix:

```bash
dtst rename -d scratch/crowd --from final/1024 --prefix "crowd_"
```

This produces `crowd_1.jpg`, `crowd_2.jpg`, etc. The number of zero-padded digits is computed automatically from the total count — 5 images get single digits, 100 images get 3 digits. To set it explicitly:

```bash
dtst rename -d scratch/crowd --from final/1024 --prefix "crowd_" --digits 5
```

This produces `crowd_00001.jpg`, `crowd_00002.jpg`, etc. Sidecar JSON files are renamed along with their images.

Preview what would happen before committing:

```bash
dtst rename -d scratch/crowd --from final/1024 --prefix "crowd_" --dry-run
```

## Format

The `format` command normalizes image formats, channels, and metadata before the final resize. This is useful when your sources contain a mix of PNG and JPEG files, images with alpha channels, or embedded EXIF data you want to strip before training.

To convert everything to JPEG and enforce RGB channels:

```bash
dtst format -d scratch/crowd --from final/1024 --to final/formatted -f jpg --channels rgb
```

To strip all EXIF metadata and ICC profiles while preserving the source format:

```bash
dtst format -d scratch/crowd --from final/1024 --to final/formatted --strip-metadata
```

You can combine multiple normalizations in a single pass — convert to WebP, enforce RGB, and strip metadata:

```bash
dtst format -d scratch/crowd --from final/1024 --to final/formatted -f webp --channels rgb --strip-metadata
```

When converting images with transparency to a format that requires a background (like JPEG), or when using `--channels rgb`, alpha channels are composited onto white by default. Use `--background` to change this:

```bash
dtst format -d scratch/crowd --from final/1024 --to final/formatted -f jpg --channels rgb --background black
```

For grayscale datasets:

```bash
dtst format -d scratch/crowd --from final/1024 --to final/formatted --channels grayscale
```

## Resize

The `frame` command resizes images to a target width and/or height using Lanczos resampling. Use it to produce sized versions of the formatted dataset:

```bash
dtst frame -d scratch/crowd --from final/formatted --to final/512 --width 512 --height 512
dtst frame -d scratch/crowd --from final/formatted --to final/256 --width 256 --height 256
```

When only one dimension is given, the other is computed proportionally to preserve the aspect ratio:

```bash
dtst frame -d scratch/crowd --from final/formatted --to final/512 --width 512
```

## Validate

The `validate` command checks that every image in a folder shares the same dimensions and channel mode. Run it against your final outputs to catch inconsistencies before training:

```bash
dtst validate -d scratch/crowd --from final/512
```

If your training target requires square images (e.g. StyleGAN), add `--square`:

```bash
dtst validate -d scratch/crowd --from final/512 --square
```

The command also warns if any PNG files use compression above level 0, which slows down data loading during training.

If everything passes you will see output like:

```
Validated 1,204 images (0m 3s)

  Dimensions: PASS (all 512x512)
  Channels:   PASS (all RGB)
  Square:     PASS
  PNG comp:   OK (all 1,204 PNGs at compression level 0)
```

If any check fails, `validate` exits with code 1 so you can use it in scripts or workflows.

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
      upscaled/          <- AI-upscaled (optional)
      formatted/         <- normalized format/channels
      1024/              <- framed (resized)
      512/               <- framed (resized)
      256/               <- framed (resized)
```

