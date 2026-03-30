import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from urllib.parse import urlparse

import click
import requests
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import FetchConfig, load_fetch_config
from dtst.throttle import DomainThrottler
from dtst.urls import canonicalize_image_url, clean_image_url
from dtst.user_agent import get_user_agent

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"})

UNSUPPORTED_EXTENSIONS = frozenset({".djvu"})

CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

MAX_RETRIES = 3

# Domains where yt-dlp should be used instead of direct HTTP download.
# These are video hosting platforms whose URLs require extraction logic.
YTDLP_DOMAINS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "vimeo.com",
    "www.vimeo.com",
    "player.vimeo.com",
    "dailymotion.com",
    "www.dailymotion.com",
    "twitch.tv",
    "www.twitch.tv",
    "clips.twitch.tv",
    "streamable.com",
    "rumble.com",
    "bitchute.com",
    "www.bitchute.com",
    "odysee.com",
    "peertube.social",
    "bilibili.com",
    "www.bilibili.com",
    "nicovideo.jp",
    "www.nicovideo.jp",
})


_YTDLP_PROGRESS_RE = re.compile(r"\[download\]\s+([\d.]+)%")


def _is_ytdlp_url(url: str) -> bool:
    """Decide whether a URL should be handled by yt-dlp.

    Returns True for known video hosting domains, False for everything
    else (including direct .mp4 links on CDNs, which requests can
    handle fine).
    """
    hostname = urlparse(url).hostname
    if hostname is None:
        return False
    hostname = hostname.lower()
    return hostname in YTDLP_DOMAINS


def _attempt_download(
    url: str, timeout: int, throttler: DomainThrottler, domain: str, max_wait: float | None,
) -> requests.Response | None:
    url_clean = clean_image_url(url)
    if url_clean == url:
        attempt_urls = (url,)
    else:
        attempt_urls = (url_clean, url)
    for attempt_url in attempt_urls:
        for retry in range(MAX_RETRIES):
            throttler.acquire(domain)
            try:
                response = requests.get(
                    attempt_url,
                    stream=True,
                    timeout=timeout,
                    headers={"User-Agent": get_user_agent()},
                )
            except requests.RequestException:
                throttler.release(domain)
                break

            if response.status_code == 200:
                throttler.record_success(domain)
                return response

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                response.close()
                throttler.release(domain)

                retry_secs = None
                if retry_after:
                    try:
                        retry_secs = float(retry_after)
                    except ValueError:
                        pass

                if retry_secs is not None and (max_wait is None or retry_secs <= max_wait):
                    backoff = retry_secs
                    logger.warning(
                        "Rate limited by %s, waiting %ds (Retry-After). Ctrl+C to abort.",
                        domain, int(backoff),
                    )
                else:
                    backoff = (1.0 * (2 ** retry)) + random.uniform(0, 1)

                logger.debug("Rate limited on %s, retrying in %.1fs (attempt %d/%d)", domain, backoff, retry + 1, MAX_RETRIES)
                time.sleep(backoff)
                continue

            response.close()
            throttler.release(domain)
            break
        else:
            throttler.release(domain)
            continue
        break

    return None


def _download_url(args: tuple) -> tuple[str, str, str | None]:
    url, dest_dir, timeout, force, throttler, max_wait = args
    url_hash = hashlib.md5(url.encode()).hexdigest()
    domain = urlparse(url).hostname or "unknown"

    try:
        if throttler.is_tripped(domain):
            return "rate_limited", url, None

        if not force:
            for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                if (dest_dir / f"{url_hash}{ext}").exists():
                    return "skipped", url, None

        tmp_path = dest_dir / f"{url_hash}.tmp"

        response = _attempt_download(url, timeout, throttler, domain, max_wait)

        if response is None:
            throttler.record_429(domain)
            return "failed", url, "all URL variants returned non-200 or failed"

        try:
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        finally:
            response.close()
            throttler.release(domain)

        content_type = response.headers.get("Content-Type", "")
        mime = content_type.split(";")[0].strip().lower()
        ext = CONTENT_TYPE_TO_EXT.get(mime)

        # For images, validate with PIL and use Content-Type extension
        if ext is not None:
            try:
                with Image.open(tmp_path) as img:
                    img.verify()
            except Exception:
                tmp_path.unlink(missing_ok=True)
                return "failed", url, "downloaded file is not a valid image"

            final_path = dest_dir / f"{url_hash}{ext}"
            tmp_path.rename(final_path)
            return "downloaded", url, None

        # For non-image content (e.g. direct .mp4 links), keep the file
        # and derive extension from the URL path or Content-Type
        url_ext = Path(urlparse(url).path).suffix.lower()
        if url_ext in VIDEO_EXTENSIONS:
            ext = url_ext
        elif "video/" in mime:
            ext = ".mp4"
        else:
            # Fallback: assume image and try to validate
            try:
                with Image.open(tmp_path) as img:
                    img.verify()
                ext = ".jpg"
            except Exception:
                tmp_path.unlink(missing_ok=True)
                return "failed", url, f"unknown content type: {mime}"

        final_path = dest_dir / f"{url_hash}{ext}"
        tmp_path.rename(final_path)
        return "downloaded", url, None

    except Exception as e:
        tmp_path = dest_dir / f"{url_hash}.tmp"
        tmp_path.unlink(missing_ok=True)
        return "failed", url, str(e)


def _download_ytdlp(args: tuple, progress_callback=None) -> tuple[str, str, str | None]:
    """Download a URL using yt-dlp as a subprocess.

    When *progress_callback* is provided it is called with
    ``(url, percentage)`` each time yt-dlp reports download progress.
    The ``--newline`` flag ensures progress lines are flushed on
    separate lines so they can be parsed in real time.
    """
    url, dest_dir_s, archive_path_s, force = args
    dest_dir = Path(dest_dir_s)

    try:
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--newline",
            "--socket-timeout", "60",
            # Video-only (no audio) with fallback to best combined
            # if separate video streams are unavailable. Prefer highest
            # resolution, best codecs (H.264 > AV1 > VP9), highest bitrate.
            "-f", "bv/b",
            "-S", "res,vcodec:h264:av01:vp9,br",
            "-o", str(dest_dir / "%(id)s.%(ext)s"),
            "--no-overwrites",
        ]

        if not force and archive_path_s:
            cmd.extend(["--download-archive", archive_path_s])

        cmd.append(url)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        # Read stderr in a daemon thread to prevent pipe buffer deadlock
        stderr_chunks: list[str] = []
        stderr_thread = threading.Thread(
            target=lambda: stderr_chunks.extend(proc.stderr), daemon=True,
        )
        stderr_thread.start()

        stdout_chunks: list[str] = []
        for line in proc.stdout:
            stdout_chunks.append(line)
            if progress_callback is not None:
                m = _YTDLP_PROGRESS_RE.search(line)
                if m:
                    progress_callback(url, float(m.group(1)))

        proc.wait()
        stderr_thread.join(timeout=10)
        stderr = "".join(stderr_chunks).strip()
        stdout_full = "".join(stdout_chunks)

        if proc.returncode == 0:
            return "downloaded", url, None

        # yt-dlp returns 0 even for "already recorded", but some
        # versions may use a non-zero exit code; treat archive
        # hits as skips
        if "has already been recorded" in stderr or "has already been recorded" in stdout_full:
            return "skipped", url, None

        return "failed", url, stderr[-200:] if len(stderr) > 200 else stderr

    except FileNotFoundError:
        return "failed", url, "yt-dlp not found (is it installed?)"
    except Exception as e:
        return "failed", url, str(e)


def _load_urls_from_jsonl(
    results_file: Path,
    min_size: int,
    license_filter: str | None,
) -> tuple[list[str], int]:
    urls: list[str] = []
    skipped_unsupported = 0
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            w = r.get("width")
            h = r.get("height")
            if w is not None and h is not None:
                try:
                    if max(int(w), int(h)) < min_size:
                        continue
                except (TypeError, ValueError):
                    pass
            if license_filter is not None:
                lic = r.get("license")
                if not lic or not lic.startswith(license_filter):
                    continue
            url = r.get("url")
            if url:
                url = canonicalize_image_url(url)
                path = urlparse(url).path
                ext = Path(path).suffix.lower()
                if ext in UNSUPPORTED_EXTENSIONS:
                    skipped_unsupported += 1
                    continue
                urls.append(url)
    return sorted(set(urls)), skipped_unsupported


def _load_urls_from_txt(txt_file: Path) -> list[str]:
    """Load URLs from a plain text file, one per line.

    Blank lines and lines starting with ``#`` are ignored.
    """
    urls: list[str] = []
    with open(txt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return sorted(set(urls))


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    to: str | None,
    input_file: str | None,
    min_size: int | None,
    license_filter: str | None,
) -> FetchConfig:
    if config is not None:
        cfg = load_fetch_config(config)
    else:
        cfg = FetchConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if to is not None:
        cfg.to = to
    if input_file is not None:
        cfg.input = input_file
    if min_size is not None:
        cfg.min_size = min_size
    if license_filter is not None:
        cfg.license = license_filter

    if cfg.to is None:
        raise click.ClickException("--to is required (or set 'fetch.to' in config)")

    return cfg


def _check_ytdlp() -> bool:
    """Return True if yt-dlp is available on PATH."""
    return shutil.which("yt-dlp") is not None


@click.command("fetch")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory where input is read from and media is written to (default: .).")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--input", "-i", "input_file", type=str, default=None, help="Input file name relative to the working directory. Supports .jsonl and .txt formats.")
@click.option("--min-size", "-s", type=int, default=None, help="Minimum image dimension in pixels; only applies to .jsonl input (default: 512).")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel download threads (default: CPU count for images, 2 for video).")
@click.option("--timeout", "-t", type=int, default=30, show_default=True, help="Per-request timeout in seconds.")
@click.option("--force", "-f", is_flag=True, help="Re-download files even if they already exist.")
@click.option("--max-wait", "-W", type=int, default=None, help="Max seconds to honor a Retry-After header (default: unlimited).")
@click.option("--no-wait", is_flag=True, help="Never wait for Retry-After headers; use fast exponential backoff instead.")
@click.option("--license", "-l", "license_filter", type=str, default=None, help="Only download images whose license starts with this prefix (e.g. 'cc'); only applies to .jsonl input.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    to: str | None,
    input_file: str | None,
    min_size: int | None,
    workers: int | None,
    timeout: int,
    force: bool,
    max_wait: int | None,
    no_wait: bool,
    license_filter: str | None,
) -> None:
    """Download images and videos from a URL list.

    Reads a URL list from the working directory specified by --input.
    Two formats are supported:

    \b
      .jsonl  JSON Lines with a "url" field per line (search output).
              Supports --min-size and --license filtering.
      .txt    Plain text with one URL per line. Lines starting with
              # are treated as comments.

    URLs are routed automatically: known video hosting domains
    (YouTube, Vimeo, etc.) are downloaded with yt-dlp, all other
    URLs are downloaded directly with HTTP requests.

    Image files are named by the MD5 hash of the URL. Video files
    are named by yt-dlp using the video ID and original extension.
    Existing files are skipped unless --force is set.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst fetch config.yaml
        dtst fetch -d ./chanterelle --to raw
        dtst fetch -d ./project --to videos --input urls.txt
        dtst fetch config.yaml --workers 16 --timeout 60
        dtst fetch config.yaml --force
        dtst fetch -d ./chanterelle --to raw --no-wait --license cc
    """
    if no_wait and max_wait is not None:
        raise click.ClickException("--no-wait and --max-wait are mutually exclusive")
    if no_wait:
        max_wait = 0

    cfg = _resolve_config(config, working_dir, to, input_file, min_size, license_filter)

    # Determine input file and format
    if cfg.input is None:
        raise click.ClickException("--input is required (or set 'fetch.input' in config)")
    input_name = cfg.input
    input_path = cfg.working_dir / input_name
    input_ext = Path(input_name).suffix.lower()

    if not input_path.exists():
        raise click.ClickException(f"Input file not found: {input_path}")

    # Load URLs based on file format
    skipped_unsupported = 0
    if input_ext == ".jsonl":
        urls, skipped_unsupported = _load_urls_from_jsonl(input_path, cfg.min_size, cfg.license)
        logger.info("Loaded URLs from %s (jsonl mode)", input_path)
    elif input_ext == ".txt":
        urls = _load_urls_from_txt(input_path)
        logger.info("Loaded URLs from %s (txt mode)", input_path)
    else:
        raise click.ClickException(
            f"Unsupported input file format: {input_ext} (expected .jsonl or .txt)"
        )

    if skipped_unsupported > 0:
        logger.info("Skipped %d URLs with unsupported format (.djvu).", skipped_unsupported)

    if not urls:
        raise click.ClickException("No URLs to fetch after filtering")

    dest_dir = cfg.working_dir / cfg.to
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Partition URLs into direct download vs yt-dlp
    direct_urls: list[str] = []
    ytdlp_urls: list[str] = []
    for url in urls:
        if _is_ytdlp_url(url):
            ytdlp_urls.append(url)
        else:
            direct_urls.append(url)

    if ytdlp_urls and not _check_ytdlp():
        raise click.ClickException(
            f"Found {len(ytdlp_urls)} video URL(s) requiring yt-dlp but it is not installed. "
            "Install it with: pip install yt-dlp"
        )

    logger.info(
        "Fetching %d URLs to %s (%d direct, %d yt-dlp)",
        len(urls), dest_dir, len(direct_urls), len(ytdlp_urls),
    )

    start_time = time.monotonic()
    downloaded = 0
    skipped = 0
    failed = 0
    rate_limited = 0
    rate_limited_domains: set[str] = set()

    # --- Direct HTTP downloads -----------------------------------------------

    if direct_urls:
        num_workers = workers if workers is not None else cpu_count() or 4
        throttler = DomainThrottler()
        for domain, policy in throttler.active_policies().items():
            logger.info(
                "Throttle policy for %s: max_connections=%d, request_delay=%.1fs",
                domain, policy.max_connections, policy.request_delay,
            )

        logger.info("Downloading %d URLs via HTTP (workers=%d)", len(direct_urls), num_workers)

        work = [(url, dest_dir, timeout, force, throttler, max_wait) for url in direct_urls]

        with logging_redirect_tqdm():
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_download_url, w): w for w in work}
                with tqdm(total=len(futures), desc="Fetching", unit="url", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]") as pbar:
                    try:
                        for future in as_completed(futures):
                            status, url, error = future.result()
                            if status == "downloaded":
                                downloaded += 1
                            elif status == "skipped":
                                skipped += 1
                            elif status == "rate_limited":
                                rate_limited += 1
                                domain = urlparse(url).hostname or "unknown"
                                rate_limited_domains.add(domain)
                            else:
                                failed += 1
                                logger.error("Failed to download %s: %s", url, error)
                            pbar.set_postfix(ok=downloaded, skip=skipped, fail=failed, limited=rate_limited)
                            pbar.update(1)
                    except KeyboardInterrupt:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

    # --- yt-dlp downloads ----------------------------------------------------

    if ytdlp_urls:
        ytdlp_workers = workers if workers is not None else 2
        archive_path = str(dest_dir / ".ytdl-archive") if not force else None

        logger.info("Downloading %d URLs via yt-dlp (workers=%d)", len(ytdlp_urls), ytdlp_workers)

        # Shared progress state updated by worker threads
        _ytdlp_progress: dict[str, float] = {}
        _progress_lock = threading.Lock()

        def _on_progress(url: str, pct: float) -> None:
            with _progress_lock:
                _ytdlp_progress[url] = pct

        work_yt = [
            (url, str(dest_dir), archive_path, force)
            for url in ytdlp_urls
        ]

        with logging_redirect_tqdm():
            with ThreadPoolExecutor(max_workers=ytdlp_workers) as executor:
                all_futures = {
                    executor.submit(_download_ytdlp, w, _on_progress): w
                    for w in work_yt
                }
                done_futures: set = set()
                total_videos = len(all_futures)
                with tqdm(
                    total=total_videos,
                    desc="Fetching (yt-dlp)",
                    unit="video",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
                ) as pbar:
                    try:
                        prev_frac = 0.0
                        while len(done_futures) < total_videos:
                            newly_done = {f for f in all_futures if f.done()} - done_futures
                            for f in newly_done:
                                done_futures.add(f)
                                status, url, error = f.result()
                                if status == "downloaded":
                                    downloaded += 1
                                elif status == "skipped":
                                    skipped += 1
                                else:
                                    failed += 1
                                    logger.error("Failed to download %s: %s", url, error)
                                with _progress_lock:
                                    _ytdlp_progress.pop(url, None)

                            # Smoothly advance the bar: completed videos
                            # count as 1.0, active ones contribute their
                            # current fraction (0..1).
                            with _progress_lock:
                                active_frac = sum(
                                    p / 100.0 for p in _ytdlp_progress.values()
                                )
                            current = len(done_futures) + active_frac
                            delta = current - prev_frac
                            if delta > 0:
                                pbar.n = int(current * 100) / 100
                                pbar.refresh()
                                prev_frac = current

                            pbar.set_postfix(ok=downloaded, skip=skipped, fail=failed)

                            if len(done_futures) < total_videos:
                                time.sleep(0.3)
                    except KeyboardInterrupt:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

    # --- Summary -------------------------------------------------------------

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nFetch complete!")
    click.echo(f"  Downloaded: {downloaded:,}")
    click.echo(f"  Skipped (existing): {skipped:,}")
    if skipped_unsupported > 0:
        click.echo(f"  Skipped (unsupported format): {skipped_unsupported:,}")
    if rate_limited > 0:
        domains_str = ", ".join(sorted(rate_limited_domains))
        click.echo(f"  Rate limited: {rate_limited:,} ({domains_str})")
    click.echo(f"  Failed: {failed:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {dest_dir}")
