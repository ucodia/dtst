# dtst

`dtst` is a Python toolkit for researching, collecting, curating, and preparing image datasets for machine learning — StyleGAN training, LoRA fine-tuning, computer vision research, and more.

It provides a complete set of independent CLI tools for every stage of dataset preparation, from initial data collection to final preprocessing. Each tool can be used standalone or combined into automated workflows.

## Features

- **Search** multiple image engines (Flickr, Brave, Google, Wikimedia) and accumulate results
- **Fetch** images and videos from URL lists, with automatic yt-dlp support for video platforms
- **Extract frames** from videos as high-quality keyframes
- **Extract faces** with aligned cropping using MediaPipe or dlib
- **Cluster** images by visual similarity (ArcFace for faces, CLIP for general images)
- **Analyze** images for perceptual hashes and blur scores
- **Filter** by size, blur, or detection scores — non-destructively
- **Review** with an interactive web UI for manual review
- **Deduplicate** by perceptual hash similarity
- **Augment** with flips and **resize** to target dimensions
- **Run** multi-step workflows from a single config file

## Install

`dtst` ships a lean core with opt-in extras — install only what the commands you use require:

```bash
uv tool install git+https://github.com/Ucodia/dtst.git                  # lean core
uv tool install "git+https://github.com/Ucodia/dtst.git#egg=dtst[all]"  # everything
```

Available extras:

| Extra    | Enables                                          | Adds                                                              |
|----------|--------------------------------------------------|-------------------------------------------------------------------|
| `faces`  | `extract-faces`                                  | insightface, mediapipe, dlib, onnxruntime, scipy                  |
| `torch`  | `analyze` (IQA), `cluster`, `detect`, `upscale`  | torch, transformers, open-clip, spandrel, pyiqa, hdbscan, sklearn |
| `server` | `review`                                         | fastapi, uvicorn                                                  |
| `all`    | all of the above                                 | everything                                                        |

Commands that need a missing extra fail fast with a message pointing to the correct install.

Copy `.env.example` to `.env` and fill in your API keys for whichever search engines you want to use:

```bash
cp .env.example .env
```

