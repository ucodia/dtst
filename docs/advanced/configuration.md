# Configuration

For repeatable pipelines, collect your parameters into a YAML config file. Commands can be invoked with just a config file, just CLI options, or both. When both are provided, CLI options override config file values.

## Config file structure

The config file has `working_dir` at the top level and parameters nested under command-specific keys:

```yaml
# crowd.yaml
working_dir: "./scratch/crowd"

search:
  terms:
    - crowd
    - street crowd
    - urban crowd
  suffixes:
    - photography
    - candid
  engines:
    - flickr
    - brave

fetch:
  to: images/search1
  min_size: 1024

extract_frames:
  from: videos
  to: images/frames
  keyframes: 3

extract_faces:
  from: images/*
  to: faces
  skip_partial: true
  engine: dlib

cluster:
  from: faces
  to: cluster
  min_cluster_size: 20
  min_samples: 5
  top: 10
  clean: true

select:
  from:
    - cluster/000
    - cluster/001
    - cluster/003
  to: curated

analyze:
  from: curated
  phash: true
  blur: true

detect:
  from: curated
  classes:
    - microphone
    - chair
  threshold: 0.2

review:
  from: curated

dedup:
  from: curated
  threshold: 8

augment:
  from: curated
  to: final/1024
  flip_x: true

rename:
  from: final/1024
  prefix: crowd_
  digits: 4

format:
  from: final/1024
  to: final/formatted
  format: jpg
  channels: rgb
  strip_metadata: true
  quality: 95

frame:
  from: final/formatted
  to: final/512
  width: 512
  height: 512

validate:
  from: final/512
  square: true
```

## Running with a config file

Pass the config file as the first argument to any command:

```bash
dtst search crowd.yaml
dtst fetch crowd.yaml
dtst extract-frames crowd.yaml
dtst extract-faces crowd.yaml
dtst cluster crowd.yaml
dtst select crowd.yaml
dtst analyze crowd.yaml
dtst detect crowd.yaml
dtst review crowd.yaml
dtst dedup crowd.yaml
dtst augment crowd.yaml
dtst rename crowd.yaml
dtst format crowd.yaml
dtst frame crowd.yaml
dtst validate crowd.yaml
```

## CLI overrides

CLI options override the corresponding config values. This is useful for one-off adjustments:

```bash
# Use dlib instead of the config's mediapipe
dtst extract-faces crowd.yaml --engine dlib

# Stricter dedup threshold than the config
dtst dedup crowd.yaml --threshold 4

# Preview any command without executing
dtst select crowd.yaml --dry-run
```

Command-specific keys use underscores (e.g. `extract_faces`, `extract_frames`, `flip_x`), matching Python parameter names. The CLI uses hyphens (e.g. `extract-faces`, `--flip-x`). See the [CLI reference](../reference/cli.md) for the complete list of options per command.
