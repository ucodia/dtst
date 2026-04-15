from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from dtst.config import (
    config_argument,
    from_dirs_option,
    to_dir_option,
    working_dir_option,
    workers_option,
)
from dtst.executor import run_pool
from dtst.files import find_images, format_elapsed, resolve_dirs, resolve_workers
from dtst.sidecar import EXCLUDE_METRICS_AND_CLASSES, copy_sidecar

logger = logging.getLogger(__name__)


def _process_image(args: tuple) -> tuple[str, str, int, str | None]:
    """Top-level worker function for ProcessPoolExecutor.

    Returns ``(status, filename, face_count, error_message)``.
    Status is one of ``"ok"``, ``"no_faces"``, ``"failed"``.
    """
    (
        input_path_s,
        output_dir_s,
        max_size,
        engine,
        max_faces,
        padding,
        skip_partial,
        refine_landmarks,
        debug,
    ) = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    name = input_path.name

    try:
        import os
        import cv2
        from dtst.face_align import FaceAligner

        image = cv2.imread(str(input_path))
        if image is None:
            return "failed", name, 0, "could not read image"

        # MediaPipe emits verbose C++ startup messages to stderr on first use.
        # Redirect fd 2 at the OS level so nothing gets through regardless of
        # how glog is configured.
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _saved_stderr = os.dup(2)
        os.dup2(_devnull, 2)
        os.close(_devnull)
        try:
            aligner = FaceAligner(
                engine=engine,
                max_faces=max_faces,
                refine_landmarks=refine_landmarks,
            )
        finally:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
        faces = aligner.get_aligned_faces(
            image,
            max_size=max_size,
            max_faces=max_faces,
            enable_padding=padding,
            skip_partial=skip_partial,
            debug=debug,
        )

        if not faces:
            return "no_faces", name, 0, None

        stem = input_path.stem
        for i, face_img in enumerate(faces):
            if len(faces) == 1:
                out_name = f"{stem}.jpg"
            else:
                out_name = f"{stem}_{i + 1:02d}.jpg"
            face_img.save(output_dir / out_name, "JPEG", quality=95)

        return "ok", name, len(faces), None

    except Exception as e:
        return "failed", name, 0, str(e)


@click.command("extract-faces")
@config_argument
@working_dir_option(
    help="Working directory containing source folders and where output is written (default: .)."
)
@from_dirs_option()
@to_dir_option()
@click.option(
    "--max-size",
    "-M",
    type=int,
    default=None,
    help="Maximum side length in pixels; faces smaller than this are kept at natural size (default: no limit).",
)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["mediapipe", "dlib"], case_sensitive=False),
    default=None,
    help="Face detection engine (default: mediapipe).",
)
@click.option(
    "--max-faces",
    "-m",
    type=int,
    default=None,
    help="Max faces to extract per image (default: 1).",
)
@workers_option()
@click.option(
    "--padding/--no-padding",
    default=None,
    help="Enable/disable reflective padding on crops (default: enabled).",
)
@click.option(
    "--skip-partial",
    is_flag=True,
    help="Skip faces whose crop extends beyond the image boundary instead of padding them.",
)
@click.option(
    "--refine-landmarks",
    is_flag=True,
    help="Enable MediaPipe refined landmarks (478 vs 468).",
)
@click.option("--debug", is_flag=True, help="Overlay landmark points on output images.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    max_size: int | None,
    engine: str | None,
    max_faces: int | None,
    workers: int | None,
    padding: bool | None,
    skip_partial: bool,
    refine_landmarks: bool,
    debug: bool,
) -> None:
    """Extract aligned face crops from images.

    Detects faces in each image using MediaPipe (default) or dlib,
    then produces an aligned and cropped face image for each detection.
    The alignment normalises eye and mouth positions for consistent
    face crops.

    Reads images from one or more source folders within the working
    directory and writes face crops to a destination folder. Multiple
    source folders can be specified as a comma-separated list with
    --from.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst extract-faces config.yaml
        dtst extract-faces config.yaml --engine dlib --max-size 512
        dtst extract-faces -d ./crowd --from raw --to faces
        dtst extract-faces -d ./crowd --from raw,extra --to faces
        dtst extract-faces config.yaml --max-faces 3 --no-padding
    """
    if not from_dirs:
        raise click.ClickException(
            "--from is required (or set 'extract_faces.from' in config)"
        )
    if not to:
        raise click.ClickException(
            "--to is required (or set 'extract_faces.to' in config)"
        )

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()
    engine = engine or "mediapipe"
    max_faces = max_faces if max_faces is not None else 1
    if padding is None:
        padding = True

    input_dirs = resolve_dirs(working, dirs_list)
    output_dir = working / to

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise click.ClickException(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise click.ClickException(
            f"No images found in: {', '.join(str(d) for d in input_dirs)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = resolve_workers(workers)

    max_size_label = str(max_size) if max_size is not None else "none"
    from_label = ", ".join(str(d) for d in input_dirs)
    logger.info(
        "Extracting faces from %d images across [%s] (engine=%s, max_size=%s, max_faces=%d, workers=%d)",
        len(images),
        from_label,
        engine,
        max_size_label,
        max_faces,
        num_workers,
    )

    work = [
        (
            str(img_path),
            str(output_dir),
            max_size,
            engine,
            max_faces,
            padding,
            skip_partial,
            refine_landmarks,
            debug,
        )
        for img_path in images
    ]

    start_time = time.monotonic()
    total_faces = 0

    def handle(result, work_item):
        nonlocal total_faces
        status, name, face_count, error = result
        if status == "ok":
            total_faces += face_count
            src_path = Path(work_item[0])
            stem = src_path.stem
            if face_count == 1:
                copy_sidecar(
                    src_path,
                    output_dir / f"{stem}.jpg",
                    exclude=EXCLUDE_METRICS_AND_CLASSES,
                )
            else:
                for i in range(face_count):
                    copy_sidecar(
                        src_path,
                        output_dir / f"{stem}_{i + 1:02d}.jpg",
                        exclude=EXCLUDE_METRICS_AND_CLASSES,
                    )
            return "ok"
        if status == "no_faces":
            logger.debug("No faces detected in %s", name)
            return "noface"
        logger.error("Failed to process %s: %s", name, error)
        return "fail"

    counts = run_pool(
        ProcessPoolExecutor,
        _process_image,
        work,
        max_workers=num_workers,
        desc="Extracting faces",
        unit="image",
        on_result=handle,
        postfix_keys=("ok", "noface", "fail"),
    )
    ok_count = counts.get("ok", 0)
    no_faces_count = counts.get("noface", 0)
    failed_count = counts.get("fail", 0)

    elapsed = time.monotonic() - start_time

    click.echo("\nExtract faces complete!")
    click.echo(f"  Processed: {ok_count:,}")
    click.echo(f"  Faces extracted: {total_faces:,}")
    click.echo(f"  No faces detected: {no_faces_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {format_elapsed(elapsed)}")
    click.echo(f"  Output: {output_dir}")
