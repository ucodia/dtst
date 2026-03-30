from __future__ import annotations

import logging
import time
from pathlib import Path

import click
import numpy as np
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import TagConfig, load_tag_config
from dtst.files import find_images, resolve_dirs
from dtst.sidecar import read_sidecar, sidecar_path, write_sidecar

logger = logging.getLogger(__name__)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    labels: str | None,
    batch_size: int | None,
) -> TagConfig:
    if config is not None:
        cfg = load_tag_config(config)
    else:
        cfg = TagConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
    if labels is not None:
        cfg.labels = [l.strip() for l in labels.split(",") if l.strip()]
    if batch_size is not None:
        cfg.batch_size = batch_size

    return cfg


@click.command("tag")
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "--from",
    "from_dirs",
    type=str,
    default=None,
    help="Comma-separated source folders (supports globs, e.g. 'images/*').",
)
@click.option(
    "--labels",
    "-l",
    type=str,
    default=None,
    help="Comma-separated text labels for zero-shot classification.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=None,
    help="Images per inference batch.",
    show_default="32",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Recompute tags even if sidecar data already exists.",
)
@click.option(
    "--working-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: .).",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of threads for image preloading (default: 4).",
)
@click.option("--clear", is_flag=True, help="Remove all tag data from sidecar files.")
@click.option("--dry-run", is_flag=True, help="Preview what would be tagged without writing sidecars.")
def cmd(config, from_dirs, labels, batch_size, force, working_dir, workers, clear, dry_run):
    """Score images against text labels using CLIP zero-shot classification.

    Computes a similarity score for each image against each text label
    and writes the results into per-image sidecar JSON files under a
    "tags" key. Scores range from -1 to 1 (higher means stronger match).

    Results are incremental — running with different label sets accumulates
    scores in the sidecar. Use --force to recompute all labels.

    \b
    Examples:
        dtst tag -d ./project --from raw --labels "microphone,photograph,illustration"
        dtst tag config.yaml
        dtst tag -d ./project --from raw --labels "cartoon,screenshot" --force
        dtst tag -d ./project --from raw --labels "microphone" --dry-run
        dtst tag -d ./project --from raw --clear
    """
    t0 = time.time()

    cfg = _resolve_config(config, working_dir, from_dirs, labels, batch_size)

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'tag.from' in config)")

    working = cfg.working_dir.resolve()
    input_dirs = resolve_dirs(working, cfg.from_dirs)

    all_images: list[Path] = []
    for src in input_dirs:
        if not src.is_dir():
            logger.warning("Source directory does not exist, skipping: %s", src)
            continue
        all_images.extend(find_images(src))

    if not all_images:
        raise click.ClickException("No images found in source directories.")

    # --- Clear mode ----------------------------------------------------------

    if clear:
        modified = 0
        for img in all_images:
            sc = sidecar_path(img)
            if not sc.exists():
                continue
            sidecar = read_sidecar(img)
            if "tags" not in sidecar:
                continue
            if dry_run:
                modified += 1
                continue
            del sidecar["tags"]
            if sidecar:
                import json

                with open(sc, "w") as f:
                    json.dump(sidecar, f, indent=2)
                    f.write("\n")
            else:
                sc.unlink()
            modified += 1

        if dry_run:
            click.echo(f"[dry-run] Would clear tag data from {modified:,} sidecar files")
        else:
            elapsed = time.time() - t0
            click.echo(f"Cleared tag data from {modified:,} sidecar files ({elapsed:.1f}s)")
        return

    # --- Tag mode ------------------------------------------------------------

    if cfg.labels is None or not cfg.labels:
        raise click.ClickException("--labels is required (or set 'tag.labels' in config)")

    label_set = set(cfg.labels)
    needs_work: list[Path] = []
    skipped = 0
    for img in all_images:
        if not force:
            sidecar = read_sidecar(img)
            existing_scores = sidecar.get("tags", {}).get("scores", {})
            if label_set.issubset(existing_scores.keys()):
                skipped += 1
                continue
        needs_work.append(img)

    if dry_run:
        click.echo(f"[dry-run] Would tag {len(needs_work):,} images with labels: {', '.join(cfg.labels)}")
        click.echo(f"  Skipping {skipped:,} images (already tagged)")
        return

    if not needs_work:
        click.echo(f"All {len(all_images):,} images already tagged with requested labels. Nothing to do.")
        return

    logger.info(
        "Tagging %d images with %d labels (%d skipped)",
        len(needs_work),
        len(cfg.labels),
        skipped,
    )

    num_workers = workers if workers is not None else 4

    from dtst.embeddings.base import detect_device
    from dtst.embeddings.clip import CLIPBackend

    device = detect_device()
    backend = CLIPBackend()
    backend.load(device)

    with logging_redirect_tqdm():
        scores, valid_paths = backend.tag(
            needs_work,
            cfg.labels,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
        )

    written = 0
    for img_path, img_scores in scores.items():
        existing = read_sidecar(img_path)
        existing_tags = existing.get("tags", {})
        existing_labels = set(existing_tags.get("labels", []))
        existing_scores = existing_tags.get("scores", {})

        merged_labels = sorted(existing_labels | set(cfg.labels))
        merged_scores = {**existing_scores, **img_scores}

        write_sidecar(img_path, {"tags": {"labels": merged_labels, "scores": merged_scores}})
        written += 1

    # Print score distribution summary
    if valid_paths:
        click.echo(f"\nScore distribution per label:")
        for label in cfg.labels:
            label_scores = [scores[p][label] for p in valid_paths if p in scores]
            if label_scores:
                arr = np.array(label_scores)
                click.echo(
                    f"  {label}: min={arr.min():.3f}  p25={np.percentile(arr, 25):.3f}  "
                    f"median={np.median(arr):.3f}  p75={np.percentile(arr, 75):.3f}  max={arr.max():.3f}"
                )

    elapsed = time.time() - t0
    click.echo(
        f"\nDone: {written:,} tagged, {skipped:,} skipped, "
        f"{len(needs_work) - len(valid_paths):,} failed ({elapsed:.1f}s)"
    )
