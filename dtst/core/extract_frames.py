"""Library-layer implementation of ``dtst extract-frames``."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.errors import InputError
from dtst.files import find_videos, resolve_dirs, resolve_workers
from dtst.results import ExtractFramesResult
from dtst.sidecar import EXCLUDE_METRICS_AND_CLASSES, copy_sidecar

logger = logging.getLogger(__name__)

_FFMPEG_TIME_RE = re.compile(r"out_time_us=(\d+)")


def _probe_duration(video_path: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _extract_frames_worker(
    args: tuple, progress_callback=None
) -> tuple[str, str, int, str | None]:
    video_path_s, output_dir_s, keyframes_interval, fmt, duration = args
    video_path = Path(video_path_s)
    output_dir = Path(output_dir_s)
    stem = video_path.stem
    name = video_path.name

    output_pattern = str(output_dir / f"{stem}_%04d.{fmt}")

    try:
        existing = list(output_dir.glob(f"{stem}_*.{fmt}"))
        if existing:
            return "skipped", name, 0, None

        select_expr = (
            f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,{keyframes_interval})"
        )

        cmd = [
            "ffmpeg",
            "-skip_frame",
            "nokey",
            "-i",
            video_path_s,
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "vfr",
            "-q:v",
            "2" if fmt == "jpg" else "0",
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            output_pattern,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stderr_chunks: list[str] = []
        stderr_thread = threading.Thread(
            target=lambda: stderr_chunks.extend(proc.stderr),
            daemon=True,
        )
        stderr_thread.start()

        duration_us = duration * 1_000_000 if duration and duration > 0 else None
        for line in proc.stdout:
            if progress_callback is not None and duration_us:
                m = _FFMPEG_TIME_RE.search(line)
                if m:
                    current_us = int(m.group(1))
                    pct = min(current_us / duration_us * 100, 100.0)
                    progress_callback(video_path_s, pct)

        proc.wait()
        stderr_thread.join(timeout=10)

        if proc.returncode != 0:
            stderr = "".join(stderr_chunks).strip()
            return "failed", name, 0, stderr[-200:] if len(stderr) > 200 else stderr

        extracted = list(output_dir.glob(f"{stem}_*.{fmt}"))
        return "ok", name, len(extracted), None

    except FileNotFoundError:
        return "failed", name, 0, "ffmpeg not found (is it installed?)"
    except Exception as e:
        return "failed", name, 0, str(e)


def _check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_frames(
    *,
    from_dirs: str,
    to: str,
    keyframes: float = 10.0,
    fmt: str = "jpg",
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> ExtractFramesResult:
    """Extract keyframes from video files using ffmpeg."""
    if keyframes <= 0:
        raise InputError("keyframes must be a positive number")
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]

    if not _check_ffmpeg():
        raise InputError(
            "ffmpeg is not installed or not on PATH. Install it with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    input_dirs = resolve_dirs(dirs_list)
    output_dir = Path(to).expanduser().resolve()

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise InputError(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    videos: list[Path] = []
    for input_dir in input_dirs:
        found = find_videos(input_dir)
        logger.info("Found %d videos in %s", len(found), input_dir)
        videos.extend(found)

    if not videos:
        raise InputError(
            f"No video files found in: {', '.join(str(d) for d in input_dirs)}"
        )

    num_workers = resolve_workers(workers)
    from_label = ", ".join(str(d) for d in input_dirs)

    logger.info(
        "Extracting keyframes from %d videos in [%s] (interval=%.1fs, format=%s, workers=%d)",
        len(videos),
        from_label,
        keyframes,
        fmt,
        num_workers,
    )

    if dry_run:
        return ExtractFramesResult(
            processed=0,
            frames_extracted=0,
            skipped=0,
            failed=0,
            output_dir=output_dir,
            dry_run=True,
            total_videos=len(videos),
            keyframes=keyframes,
            fmt=fmt,
            elapsed=0.0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Probing video durations...")
    durations: dict[str, float | None] = {}
    for video_path in videos:
        durations[str(video_path)] = _probe_duration(str(video_path))

    work = [
        (
            str(video_path),
            str(output_dir),
            keyframes,
            fmt,
            durations.get(str(video_path)),
        )
        for video_path in videos
    ]

    _progress: dict[str, float] = {}
    _progress_lock = threading.Lock()

    def _on_progress(video_path_s: str, pct: float) -> None:
        with _progress_lock:
            _progress[video_path_s] = pct

    start_time = time.monotonic()
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    total_frames = 0

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            all_futures = {
                executor.submit(_extract_frames_worker, w, _on_progress): w
                for w in work
            }
            done_futures: set = set()
            total_videos = len(all_futures)
            with tqdm(
                total=total_videos,
                desc="Extracting keyframes",
                unit="video",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
                disable=not progress,
            ) as pbar:
                try:
                    prev_frac = 0.0
                    while len(done_futures) < total_videos:
                        newly_done = {f for f in all_futures if f.done()} - done_futures
                        for f in newly_done:
                            done_futures.add(f)
                            status, name, frame_count, error = f.result()
                            video_path_s = all_futures[f][0]
                            if status == "ok":
                                ok_count += 1
                                total_frames += frame_count
                                video_path = Path(video_path_s)
                                stem = video_path.stem
                                for frame_path in sorted(
                                    output_dir.glob(f"{stem}_*.{fmt}")
                                ):
                                    copy_sidecar(
                                        video_path,
                                        frame_path,
                                        exclude=EXCLUDE_METRICS_AND_CLASSES,
                                    )
                            elif status == "skipped":
                                skipped_count += 1
                            else:
                                failed_count += 1
                                logger.error(
                                    "Failed to extract frames from %s: %s", name, error
                                )
                            with _progress_lock:
                                _progress.pop(video_path_s, None)
                            pbar.update(0)

                        with _progress_lock:
                            active_frac = sum(p / 100.0 for p in _progress.values())
                        current = len(done_futures) + active_frac
                        delta = current - prev_frac
                        if delta > 0:
                            pbar.n = int(current * 100) / 100
                            pbar.refresh()
                            prev_frac = current

                        pbar.set_postfix(
                            ok=ok_count,
                            skip=skipped_count,
                            fail=failed_count,
                            frames=total_frames,
                        )

                        if len(done_futures) < total_videos:
                            time.sleep(0.3)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    return ExtractFramesResult(
        processed=ok_count,
        frames_extracted=total_frames,
        skipped=skipped_count,
        failed=failed_count,
        output_dir=output_dir,
        dry_run=False,
        total_videos=len(videos),
        keyframes=keyframes,
        fmt=fmt,
        elapsed=time.monotonic() - start_time,
    )
