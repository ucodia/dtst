"""Library-layer implementation of ``dtst extract-faces``."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import find_images, resolve_dirs, resolve_workers
from dtst.results import ExtractFacesResult
from dtst.sidecar import EXCLUDE_METRICS_AND_CLASSES, copy_sidecar

logger = logging.getLogger(__name__)


def _process_image(args: tuple) -> tuple[str, str, int, str | None]:
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


def extract_faces(
    *,
    working_dir: Path | None,
    from_dirs: str,
    to: str,
    max_size: int | None = None,
    engine: str = "mediapipe",
    max_faces: int = 1,
    workers: int | None = None,
    padding: bool = True,
    skip_partial: bool = False,
    refine_landmarks: bool = False,
    debug: bool = False,
    progress: bool = True,
) -> ExtractFacesResult:
    """Detect and align face crops from images."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()

    input_dirs = resolve_dirs(working, dirs_list)
    output_dir = working / to

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise InputError(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise InputError(f"No images found in: {', '.join(str(d) for d in input_dirs)}")

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
        progress=progress,
    )

    return ExtractFacesResult(
        processed=counts.get("ok", 0),
        faces_extracted=total_faces,
        no_faces=counts.get("noface", 0),
        failed=counts.get("fail", 0),
        output_dir=output_dir,
        elapsed=time.monotonic() - start_time,
    )
