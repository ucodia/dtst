import hashlib
import logging
import random
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

from dtst.config import load_config
from dtst.throttle import DomainThrottler

logger = logging.getLogger(__name__)

USER_AGENT = "dtst/0.1 (image dataset toolkit; +https://github.com/ucodia/dtst)"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})

CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

MAX_RETRIES = 3


def _attempt_download(
    url: str, timeout: int, throttler: DomainThrottler, domain: str, max_wait: float | None,
) -> requests.Response | None:
    url_clean = url.split("?")[0]
    for attempt_url in (url_clean, url):
        for retry in range(MAX_RETRIES):
            throttler.acquire(domain)
            try:
                response = requests.get(
                    attempt_url,
                    stream=True,
                    timeout=timeout,
                    headers={"User-Agent": USER_AGENT},
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
    url, raw_dir, timeout, force, throttler, max_wait = args
    url_hash = hashlib.md5(url.encode()).hexdigest()
    domain = urlparse(url).hostname or "unknown"

    try:
        if throttler.is_tripped(domain):
            return "rate_limited", url, None

        if not force:
            for ext in IMAGE_EXTENSIONS:
                if (raw_dir / f"{url_hash}{ext}").exists():
                    return "skipped", url, None

        tmp_path = raw_dir / f"{url_hash}.tmp"

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
        ext = CONTENT_TYPE_TO_EXT.get(mime, ".jpg")

        try:
            with Image.open(tmp_path) as img:
                img.verify()
        except Exception:
            tmp_path.unlink(missing_ok=True)
            return "failed", url, "downloaded file is not a valid image"

        final_path = raw_dir / f"{url_hash}{ext}"
        tmp_path.rename(final_path)
        return "downloaded", url, None

    except Exception as e:
        tmp_path = raw_dir / f"{url_hash}.tmp"
        tmp_path.unlink(missing_ok=True)
        return "failed", url, str(e)


@click.command("fetch")
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel download threads (default: CPU count).")
@click.option("--timeout", "-t", type=int, default=30, show_default=True, help="Per-request timeout in seconds.")
@click.option("--force", "-f", is_flag=True, help="Re-download files even if they already exist.")
@click.option("--max-wait", "-W", type=int, default=None, help="Max seconds to honor a Retry-After header (default: unlimited).")
@click.option("--no-wait", is_flag=True, help="Never wait for Retry-After headers; use fast exponential backoff instead.")
def cmd(config: Path, workers: int | None, timeout: int, force: bool, max_wait: int | None, no_wait: bool) -> None:
    """Download images from a urls.txt file.

    Reads the output_dir from a subject YAML config, loads urls.txt
    from that directory, and downloads each URL into a raw/
    subdirectory. Files are named by the MD5 hash of the URL with the
    extension derived from the HTTP Content-Type header. Existing files
    are skipped unless --force is set.

    Per-domain throttling is applied automatically to respect server
    rate limits (e.g. Wikimedia allows max 2 concurrent connections).
    If a domain returns repeated 429 errors, remaining URLs for that
    domain are skipped.

    By default, Retry-After headers are honored. Use --max-wait to cap
    the wait time, or --no-wait to skip waiting entirely. The two flags
    are mutually exclusive.

    \b
    Examples:

        dtst fetch subjects/trump.yaml
        dtst fetch subjects/trump.yaml --workers 16 --timeout 60
        dtst fetch subjects/trump.yaml --force
        dtst fetch subjects/trump.yaml --max-wait 30
        dtst fetch subjects/trump.yaml --no-wait
    """
    if no_wait and max_wait is not None:
        raise click.ClickException("--no-wait and --max-wait are mutually exclusive")
    if no_wait:
        max_wait = 0

    cfg = load_config(config)
    output_dir = cfg.output_dir
    urls_file = output_dir / "urls.txt"
    raw_dir = output_dir / "raw"

    if not urls_file.exists():
        raise click.ClickException(f"URLs file not found: {urls_file}")

    with open(urls_file) as f:
        urls = sorted({line.strip() for line in f if line.strip()})

    if not urls:
        raise click.ClickException(f"No URLs found in {urls_file}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    num_workers = workers if workers is not None else cpu_count() or 4

    throttler = DomainThrottler()
    for domain, policy in throttler.active_policies().items():
        logger.info("Throttle policy for %s: max_connections=%d, request_delay=%.1fs", domain, policy.max_connections, policy.request_delay)

    logger.info("Fetching %d URLs to %s (workers=%d)", len(urls), raw_dir, num_workers)

    start_time = time.monotonic()
    downloaded = 0
    skipped = 0
    failed = 0
    rate_limited = 0
    rate_limited_domains: set[str] = set()

    work = [(url, raw_dir, timeout, force, throttler, max_wait) for url in urls]

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_download_url, w): w for w in work}
            with tqdm(total=len(futures), desc="Fetching", unit="url") as pbar:
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

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nFetch complete!")
    click.echo(f"  Downloaded: {downloaded:,}")
    click.echo(f"  Skipped (existing): {skipped:,}")
    if rate_limited > 0:
        domains_str = ", ".join(sorted(rate_limited_domains))
        click.echo(f"  Rate limited: {rate_limited:,} ({domains_str})")
    click.echo(f"  Failed: {failed:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {raw_dir}")
