<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo-light.svg" alt="dtst" width="280">
  </picture>
</p>

# dtst

`dtst` is a Python toolkit for researching, collecting, curating, and preparing image datasets for machine learning applications such as StyleGAN training and computer vision research.

This toolkit provides a complete set of independent tools for dataset preparation, from initial data collection to final preprocessing. Each tool can be used standalone or combined in custom workflows based on your specific needs.

## Install

`dtst` ships a lean core and opt-in extras so you only pay for the stack you need:

```bash
pip install dtst                    # lean: search, fetch, dedup, format, frame, select, ...
pip install "dtst[faces]"           # + extract-faces (mediapipe, dlib, insightface, onnxruntime)
pip install "dtst[torch]"           # + analyze (IQA), cluster, detect, upscale (torch, transformers, ...)
pip install "dtst[server]"          # + review web UI (fastapi, uvicorn)
pip install "dtst[all]"             # everything
```

Commands that need a missing extra fail early with a message pointing to the correct install command.