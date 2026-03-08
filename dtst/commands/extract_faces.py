from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import ExtractFacesConfig, load_extract_faces_config
from dtst.images import find_images

logger = logging.getLogger(__name__)


def _process_image(args: tuple) -> tuple[str, str, int, str | None]:
    """Top-level worker function for ProcessPoolExecutor.

    Returns ``(status, filename, face_count, error_message)``.
    Status is one of ``"ok"``, ``"no_faces"``, ``"failed"``.
    """
    input_path_s, output_dir_s, max_size, engine, max_faces, padding, refine_landmarks, no_stretch, debug = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    name = input_path.name

    try:
        import cv2

        from dtst.face_align import FaceAligner

        image = cv2.imread(str(input_path))
        if image is None:
            return "failed", name, 0, "could not read image"

        aligner = FaceAligner(
            engine=engine,
            max_faces=max_faces,
            refine_landmarks=refine_landmarks,
        )
        faces = aligner.get_aligned_faces(
            image,
            max_size=max_size,
            max_faces=max_faces,
            enable_padding=padding,
            debug=debug,
            no_stretch=no_stretch,
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


def _resolve_config(
    config: Path | None,
    input_dir: Path | None,
    output_dir: Path | None,
    max_size: int | None,
    engine: str | None,
    max_faces: int | None,
    padding: bool | None,
    refine_landmarks: bool,
    no_stretch: bool,
    debug: bool,
) -> ExtractFacesConfig:
    if config is not None:
        cfg = load_extract_faces_config(config)
    else:
        cfg = ExtractFacesConfig()

    if input_dir is not None:
        cfg.input_dir = input_dir
    if output_dir is not None:
        cfg.output_dir = output_dir
    if max_size is not None:
        cfg.max_size = max_size
    if engine is not None:
        cfg.engine = engine
    if max_faces is not None:
        cfg.max_faces = max_faces
    if padding is not None:
        cfg.padding = padding
    if refine_landmarks:
        cfg.refine_landmarks = True
    if no_stretch:
        cfg.no_stretch = True
    if debug:
        cfg.debug = True

    return cfg


@click.command("extract-faces")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--input-dir", "-i", type=click.Path(exists=True, path_type=Path), default=None, help="Input directory with images (default: <output_dir>/raw).")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None, help="Output directory for face crops (default: <output_dir>/faces).")
@click.option("--max-size", "-M", type=int, default=None, help="Maximum side length in pixels; faces smaller than this are kept at natural size (default: no limit).")
@click.option("--engine", "-e", type=click.Choice(["mediapipe", "dlib"], case_sensitive=False), default=None, help="Face detection engine (default: mediapipe).")
@click.option("--max-faces", "-m", type=int, default=None, help="Max faces to extract per image (default: 1).")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--padding/--no-padding", default=None, help="Enable/disable reflective padding on crops (default: enabled).")
@click.option("--refine-landmarks", is_flag=True, help="Enable MediaPipe refined landmarks (478 vs 468).")
@click.option("--no-stretch", is_flag=True, help="Prevent upscaling small faces to output size.")
@click.option("--debug", is_flag=True, help="Overlay landmark points on output images.")
def cmd(
    config: Path | None,
    input_dir: Path | None,
    output_dir: Path | None,
    max_size: int | None,
    engine: str | None,
    max_faces: int | None,
    workers: int | None,
    padding: bool | None,
    refine_landmarks: bool,
    no_stretch: bool,
    debug: bool,
) -> None:
    """Extract aligned face crops from images.

    Detects faces in each image using MediaPipe (default) or dlib,
    then produces an aligned and cropped face image for each detection.
    The alignment normalises eye and mouth positions using the FFHQ
    alignment technique.

    By default reads images from <output_dir>/raw and writes face crops
    to <output_dir>/faces. Both directories can be overridden with
    --input-dir and --output-dir.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:
        dtst extract-faces config.yaml
        dtst extract-faces config.yaml --engine dlib --max-size 512
        dtst extract-faces --input-dir ./images --output-dir ./faces
        dtst extract-faces config.yaml --max-faces 3 --no-padding
    """
    cfg = _resolve_config(
        config, input_dir, output_dir, max_size, engine, max_faces, padding,
        refine_landmarks, no_stretch, debug,
    )

    if cfg.input_dir is None:
        raise click.ClickException(
            "Input directory not specified. Use --input-dir or provide a config file with output_dir."
        )
    if cfg.output_dir is None:
        raise click.ClickException(
            "Output directory not specified. Use --output-dir or provide a config file with output_dir."
        )

    if not cfg.input_dir.is_dir():
        raise click.ClickException(f"Input directory does not exist: {cfg.input_dir}")

    images = find_images(cfg.input_dir)
    if not images:
        raise click.ClickException(f"No images found in {cfg.input_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = workers if workers is not None else cpu_count() or 4

    max_size_label = str(cfg.max_size) if cfg.max_size is not None else "none"
    logger.info(
        "Extracting faces from %d images in %s (engine=%s, max_size=%s, max_faces=%d, workers=%d)",
        len(images), cfg.input_dir, cfg.engine, max_size_label, cfg.max_faces, num_workers,
    )

    work = [
        (
            str(img_path),
            str(cfg.output_dir),
            cfg.max_size,
            cfg.engine,
            cfg.max_faces,
            cfg.padding,
            cfg.refine_landmarks,
            cfg.no_stretch,
            cfg.debug,
        )
        for img_path in images
    ]

    start_time = time.monotonic()
    ok_count = 0
    no_faces_count = 0
    failed_count = 0
    total_faces = 0

    with logging_redirect_tqdm():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_image, w): w for w in work}
            with tqdm(total=len(futures), desc="Extracting faces", unit="image") as pbar:
                try:
                    for future in as_completed(futures):
                        status, name, face_count, error = future.result()
                        if status == "ok":
                            ok_count += 1
                            total_faces += face_count
                        elif status == "no_faces":
                            no_faces_count += 1
                            logger.debug("No faces detected in %s", name)
                        else:
                            failed_count += 1
                            logger.error("Failed to process %s: %s", name, error)
                        pbar.set_postfix(ok=ok_count, noface=no_faces_count, fail=failed_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nExtract faces complete!")
    click.echo(f"  Processed: {ok_count:,}")
    click.echo(f"  Faces extracted: {total_faces:,}")
    click.echo(f"  No faces detected: {no_faces_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {cfg.output_dir}")
