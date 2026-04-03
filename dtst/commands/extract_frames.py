from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import VALID_FRAME_FORMATS, ExtractFramesConfig, load_extract_frames_config
from dtst.files import find_videos, resolve_dirs
from dtst.sidecar import copy_sidecar

logger = logging.getLogger(__name__)

_FFMPEG_TIME_RE = re.compile(r"out_time_us=(\d+)")


def _probe_duration(video_path: str) -> float | None:
    """Return video duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
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


def _extract_frames(args: tuple, progress_callback=None) -> tuple[str, str, int, str | None]:
    """Extract keyframes from a single video using ffmpeg.

    Only I-frames (keyframes) are decoded, and a minimum interval
    of *keyframes* seconds is enforced between extracted frames.

    When *progress_callback* is provided it is called with
    ``(video_path, percentage)`` each time ffmpeg reports encoding
    progress via ``-progress pipe:1``.

    Returns ``(status, filename, frame_count, error_message)``.
    Status is one of ``"ok"``, ``"skipped"``, or ``"failed"``.
    """
    video_path_s, output_dir_s, keyframes_interval, fmt, duration = args
    video_path = Path(video_path_s)
    output_dir = Path(output_dir_s)
    stem = video_path.stem
    name = video_path.name

    # Build output pattern: {stem}_{frame_number:04d}.{format}
    output_pattern = str(output_dir / f"{stem}_%04d.{fmt}")

    try:
        # Check if any frames already exist for this video
        existing = list(output_dir.glob(f"{stem}_*.{fmt}"))
        if existing:
            return "skipped", name, 0, None

        # Extract only keyframes with a minimum interval between them.
        # -skip_frame nokey: decode only I-frames (keyframes)
        # -vsync vfr: variable frame rate output (no duplicate frames)
        # select filter: keep the first frame (prev_selected_t is NaN)
        #   plus any frame at least N seconds after the last selected one
        select_expr = f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,{keyframes_interval})"

        cmd = [
            "ffmpeg",
            "-skip_frame", "nokey",
            "-i", video_path_s,
            "-vf", f"select='{select_expr}'",
            "-vsync", "vfr",
            "-q:v", "2" if fmt == "jpg" else "0",
            "-y",
            "-progress", "pipe:1",
            "-nostats",
            output_pattern,
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        # Read stderr in a daemon thread to prevent pipe buffer deadlock
        stderr_chunks: list[str] = []
        stderr_thread = threading.Thread(
            target=lambda: stderr_chunks.extend(proc.stderr), daemon=True,
        )
        stderr_thread.start()

        # Parse structured progress from stdout
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

        # Count how many frames were actually written
        extracted = list(output_dir.glob(f"{stem}_*.{fmt}"))
        return "ok", name, len(extracted), None

    except FileNotFoundError:
        return "failed", name, 0, "ffmpeg not found (is it installed?)"
    except Exception as e:
        return "failed", name, 0, str(e)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: list[str] | None,
    to: str | None,
    keyframes: float | None,
    fmt: str | None,
) -> ExtractFramesConfig:
    if config is not None:
        cfg = load_extract_frames_config(config)
    else:
        cfg = ExtractFramesConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = from_dirs
    if to is not None:
        cfg.to = to
    if keyframes is not None:
        cfg.keyframes = keyframes
    if fmt is not None:
        cfg.format = fmt

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'extract_frames.from' in config)")
    if cfg.to is None:
        raise click.ClickException("--to is required (or set 'extract_frames.to' in config)")

    return cfg


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


@click.command("extract-frames")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs, e.g. 'images/*').")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--keyframes", "-k", type=float, default=None, help="Minimum interval in seconds between extracted keyframes. Only I-frames are considered; frames closer together than this value are skipped (default: 10).")
@click.option("--format", "-F", "fmt", type=click.Choice(sorted(VALID_FRAME_FORMATS), case_sensitive=False), default=None, help="Output image format (default: jpg).")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--dry-run", is_flag=True, help="Preview what would be done without extracting frames.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    keyframes: float | None,
    fmt: str | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Extract keyframes from video files using ffmpeg.

    Reads video files from one or more source folders and extracts
    keyframes (I-frames) to a destination folder. Each video produces
    a set of numbered images named as
    ``{video_stem}_{frame_number}.{format}``.

    Only I-frames are decoded, which avoids interpolated or blurry
    frames and produces the sharpest possible output. The --keyframes
    option sets the minimum interval between extracted frames: with
    the default of 10, at most one keyframe every 10 seconds is kept.
    Lower values produce more frames, higher values produce fewer.

    Supported video formats: .mp4, .mkv, .avi, .mov, .webm, .flv,
    .wmv, .m4v.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst extract-frames -d ./project --from videos --to frames
        dtst extract-frames -d ./project --from videos --to frames --keyframes 5
        dtst extract-frames -d ./project --from videos --to frames --keyframes 30 --format png
        dtst extract-frames config.yaml
        dtst extract-frames config.yaml --keyframes 20 --dry-run
    """
    if keyframes is not None and keyframes <= 0:
        raise click.ClickException("--keyframes must be a positive number")

    parsed_from_dirs: list[str] | None = None
    if from_dirs is not None:
        parsed_from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
        if not parsed_from_dirs:
            raise click.ClickException("--from must contain at least one folder name")

    cfg = _resolve_config(config, working_dir, parsed_from_dirs, to, keyframes, fmt)

    if not _check_ffmpeg():
        raise click.ClickException(
            "ffmpeg is not installed or not on PATH. Install it with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    input_dirs = resolve_dirs(cfg.working_dir, cfg.from_dirs)
    output_dir = cfg.working_dir / cfg.to

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise click.ClickException(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    videos: list[Path] = []
    for input_dir in input_dirs:
        found = find_videos(input_dir)
        logger.info("Found %d videos in %s", len(found), input_dir)
        videos.extend(found)

    if not videos:
        raise click.ClickException(
            f"No video files found in: {', '.join(str(d) for d in input_dirs)}"
        )

    num_workers = workers if workers is not None else cpu_count() or 4
    from_label = ", ".join(str(d) for d in input_dirs)

    logger.info(
        "Extracting keyframes from %d videos in [%s] (interval=%.1fs, format=%s, workers=%d)",
        len(videos), from_label, cfg.keyframes, cfg.format, num_workers,
    )

    if dry_run:
        click.echo(f"\nDry run -- would extract keyframes from {len(videos):,} videos")
        click.echo(f"  Min interval: {cfg.keyframes}s")
        click.echo(f"  Format: {cfg.format}")
        click.echo(f"  Output: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe durations upfront so we can compute per-video progress
    logger.info("Probing video durations...")
    durations: dict[str, float | None] = {}
    for video_path in videos:
        durations[str(video_path)] = _probe_duration(str(video_path))

    work = [
        (str(video_path), str(output_dir), cfg.keyframes, cfg.format, durations.get(str(video_path)))
        for video_path in videos
    ]

    # Shared progress state updated by worker threads
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
                executor.submit(_extract_frames, w, _on_progress): w
                for w in work
            }
            done_futures: set = set()
            total_videos = len(all_futures)
            with tqdm(
                total=total_videos,
                desc="Extracting keyframes",
                unit="video",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
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
                                for frame_path in sorted(output_dir.glob(f"{stem}_*.{cfg.format}")):
                                    copy_sidecar(video_path, frame_path, exclude={"metrics", "classes"})
                            elif status == "skipped":
                                skipped_count += 1
                            else:
                                failed_count += 1
                                logger.error("Failed to extract frames from %s: %s", name, error)
                            with _progress_lock:
                                _progress.pop(video_path_s, None)
                            pbar.update(0)  # force refresh after postfix update

                        # Smoothly advance the bar: completed videos
                        # count as 1.0, active ones contribute their
                        # current fraction (0..1).
                        with _progress_lock:
                            active_frac = sum(
                                p / 100.0 for p in _progress.values()
                            )
                        current = len(done_futures) + active_frac
                        delta = current - prev_frac
                        if delta > 0:
                            pbar.n = int(current * 100) / 100
                            pbar.refresh()
                            prev_frac = current

                        pbar.set_postfix(
                            ok=ok_count, skip=skipped_count, fail=failed_count, frames=total_frames,
                        )

                        if len(done_futures) < total_videos:
                            time.sleep(0.3)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nExtract-frames complete!")
    click.echo(f"  Processed: {ok_count:,} videos")
    click.echo(f"  Frames extracted: {total_frames:,}")
    click.echo(f"  Skipped (existing): {skipped_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
