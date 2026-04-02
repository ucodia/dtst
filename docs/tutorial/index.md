# Tutorial

This tutorial walks you through building a curated image dataset from scratch. By the end you will have searched for images, downloaded them alongside video footage, clustered by similarity, refined quality, and produced final resized outputs — all within a single working directory.

The example uses "crowd" as the subject. Along the way the tutorial shows two paths through the pipeline: one that extracts and clusters faces by identity (using ArcFace), and one that clusters images directly by visual similarity (using CLIP). Pick whichever fits your use case, or combine both.

## What you will build

A curated image dataset starting from the search term "crowd". The pipeline uses 14 tools:

1. **Search** for images across multiple engines
2. **Fetch** images and videos from the collected URLs
3. **Extract frames** from downloaded videos
4. **Extract faces** from all collected images *(optional — for face datasets)*
5. **Cluster** by similarity (ArcFace for faces, CLIP for general images)
6. **Copy** selected clusters into a curation folder
7. **Analyze** images for metadata (hashes, blur scores)
8. **Tag** images with CLIP labels
9. **Filter** out low-quality images
10. **Review** manually with the web UI
11. **Dedup** to remove near-duplicates
12. **Augment** with transformations to increase diversity
13. **Upscale** images to higher resolution
14. **Frame** (resize) for the final dataset

Steps 1–3 are covered in [Collecting images](collecting-images.md), steps 4–5 in [Extracting features](extracting-features.md), steps 6–11 in [Selecting and refining](selecting-and-refining.md), and steps 12–14 in [Final preparation](final-preparation.md).

Every `dtst` command reads from and writes to **buckets** and tracks metadata in **sidecars**. See [Concepts](../concepts.md) for details on both.

## The directory structure

As you work through the steps, your working directory grows into this layout:

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
      filtered/
      rejected/
      duplicated/
    final/
      1024/
      512/
      256/
```

## Prerequisites

Install `dtst`:

```bash
uv tool install git+https://github.com/Ucodia/dtst.git
```

Copy `.env.example` to `.env` and fill in the API keys for whichever search engines you want to use:

```bash
cp .env.example .env
```

## Create a working directory

Every `dtst` pipeline lives in a working directory. Create one for this tutorial:

```bash
mkdir -p scratch/crowd
```

All the commands in the following pages use `-d scratch/crowd` to point at this folder. You can also set `working_dir` in a config file to avoid repeating it (see [Configuration](../advanced/configuration.md)).
