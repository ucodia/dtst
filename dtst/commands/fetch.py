import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
import requests
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import load_config

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


def _download_url(args: tuple[str, Path, int, bool]) -> tuple[str, str, str | None]:
    url, raw_dir, timeout, force = args
    url_hash = hashlib.md5(url.encode()).hexdigest()

    try:
        if not force:
            for ext in IMAGE_EXTENSIONS:
                if (raw_dir / f"{url_hash}{ext}").exists():
                    return "skipped", url, None

        tmp_path = raw_dir / f"{url_hash}.tmp"

        url_clean = url.split("?")[0]
        response = None
        for attempt_url in (url_clean, url):
            try:
                response = requests.get(
                    attempt_url,
                    stream=True,
                    timeout=timeout,
                    headers={"User-Agent": USER_AGENT},
                )
                if response.status_code == 200:
                    break
                response.close()
                response = None
            except requests.RequestException:
                if response is not None:
                    response.close()
                response = None

        if response is None:
            return "failed", url, "all URL variants returned non-200 or failed"

        try:
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        finally:
            response.close()

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
def cmd(config: Path, workers: int | None, timeout: int, force: bool) -> None:
    """Download images from a urls.txt file.

    Reads the output_dir from a subject YAML config, loads urls.txt
    from that directory, and downloads each URL into a raw/
    subdirectory. Files are named by the MD5 hash of the URL with the
    extension derived from the HTTP Content-Type header. Existing files
    are skipped unless --force is set.

    \b
    Examples:

        dtst fetch subjects/trump.yaml
        dtst fetch subjects/trump.yaml --workers 16 --timeout 60
        dtst fetch subjects/trump.yaml --force
    """
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

    logger.info("Fetching %d URLs to %s (workers=%d)", len(urls), raw_dir, num_workers)

    downloaded = 0
    skipped = 0
    failed = 0

    work = [(url, raw_dir, timeout, force) for url in urls]

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
                    else:
                        failed += 1
                        logger.error("Failed to download %s: %s", url, error)
                    pbar.set_postfix(ok=downloaded, skip=skipped, fail=failed)
                    pbar.update(1)

    click.echo(f"\nFetch complete!")
    click.echo(f"  Downloaded: {downloaded:,}")
    click.echo(f"  Skipped (existing): {skipped:,}")
    click.echo(f"  Failed: {failed:,}")
    click.echo(f"  Output: {raw_dir}")
