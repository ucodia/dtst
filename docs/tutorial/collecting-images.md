# Collecting images

This step covers sourcing raw material: searching for image URLs, downloading images and videos, and extracting frames from video files.

## Search

The `search` command queries one or more image search engines and writes the results to `results.jsonl` in your working directory. Nothing is downloaded yet — this step just collects URLs.

```bash
dtst search -d scratch/crowd \
  --terms "crowd,street crowd,urban crowd" \
  --suffixes "photography,candid" \
  --engines flickr,brave
```

You can re-run this command as many times as you like. Results accumulate and are deduplicated automatically, so adding more engines or terms later extends the file without creating duplicates.

To preview the query matrix without running any searches:

```bash
dtst search -d scratch/crowd \
  --terms "crowd,street crowd" \
  --suffixes "photography,candid" \
  --engines flickr \
  --dry-run
```

## Fetch

Once you have a `results.jsonl`, the `fetch` command downloads each image into a folder inside your working directory.

```bash
dtst fetch -d scratch/crowd --to images/search1
```

After this runs, `scratch/crowd/images/search1/` contains all the successfully downloaded images. Files are named by a hash of their source URL, so re-running fetch is safe — already-downloaded images are skipped.

You can filter downloads by minimum image size or license:

```bash
dtst fetch -d scratch/crowd --to images/search1 --min-size 1024 --license cc
```

### Running a second search

To expand the dataset, run another search with different terms and fetch into a separate folder:

```bash
dtst search -d scratch/crowd \
  --terms "crowd scene,festival crowd" \
  --suffixes "photo,wide angle" \
  --engines flickr,brave

dtst fetch -d scratch/crowd --to images/search2
```

Keeping separate folders per search round makes it easy to track where images came from.

### Fetching videos

You can also provide a plain text file of URLs with `--input`. This is useful for video content — URLs pointing to known video platforms (YouTube, Vimeo, etc.) are automatically downloaded with yt-dlp:

```bash
dtst fetch -d scratch/crowd --to videos --input urls.txt
```

Where `urls.txt` contains one URL per line:

```text
https://www.youtube.com/watch?v=example1
https://www.youtube.com/watch?v=example2
https://vimeo.com/example3
```

Both image and video URLs can be mixed in the same file. Videos are saved to the destination folder alongside any images.

### Adding images manually

Because the working directory is the source of truth, you can supplement fetched images by simply copying files into any folder. Drop additional crowd photos into `scratch/crowd/images/search1/` or create a new folder like `scratch/crowd/images/extra/`. No registration or import command is needed.

## Extract frames

The `extract-frames` command extracts keyframes (I-frames) from video files using ffmpeg. Only I-frames are decoded, which avoids interpolated or blurry frames and produces the sharpest possible output.

```bash
dtst extract-frames -d scratch/crowd --from videos --to images/frames
```

For denser extraction (one keyframe every 3 seconds):

```bash
dtst extract-frames -d scratch/crowd --from videos --to images/frames --keyframes 3
```

Frames are named as `{video_stem}_{frame_number}.jpg` by default. To use PNG instead:

```bash
dtst extract-frames -d scratch/crowd --from videos --to images/frames --format png
```

## Directory so far

After completing these steps, your working directory looks like this:

```
scratch/
  crowd/
    results.jsonl         <- accumulated search results
    images/
      search1/            <- images from first search
      search2/            <- images from second search
      frames/             <- keyframes extracted from videos
    videos/               <- downloaded video files
```

