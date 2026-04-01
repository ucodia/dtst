# Workflows

Instead of running each stage manually, you can define named workflows in your config file and execute them with a single command using `dtst run`.

## Defining workflows

Add a `workflows` section to your YAML config. Each workflow is a named list of steps:

```yaml
# crowd.yaml
working_dir: "./scratch/crowd"

search:
  terms: [crowd, street crowd]
  suffixes: [photography, candid]
  engines: [flickr, brave]

fetch:
  to: images/search1
  min_size: 1024

extract_faces:
  from: images/*
  to: faces

analyze:
  from: curated
  phash: true
  blur: true

select:
  from: faces
  to: curated

dedup:
  from: curated

workflows:
  collect:
    - search
    - fetch
    - extract-faces

  refine:
    - select
    - analyze
    - dedup
```

## Running a workflow

```bash
dtst run collect crowd.yaml
dtst run refine crowd.yaml
```

Each step inherits its defaults from the corresponding config section.

## Overriding step parameters

Individual steps can override their config section values inline:

```yaml
workflows:
  faces_only:
    - extract-faces:
        from: [images/search1]
        to: faces_fresh
    - analyze
    - dedup:
        threshold: 4
```

To ignore the config section entirely and start from scratch, set `inherit: false`:

```yaml
extract_faces:
  from: images/*
  to: faces
  engine: mediapipe
  max_faces: 1

workflows:
  dlib_faces:
    - extract-faces:
        inherit: false
        from: [images/search1]
        to: faces_dlib
        engine: dlib
```

## Shell commands

Shell commands can be included with `exec`:

```yaml
workflows:
  with_pause:
    - search
    - exec: "sleep 10"
    - fetch
```

## Dry run

To preview what a workflow will do without executing:

```bash
dtst run collect crowd.yaml --dry-run
```
