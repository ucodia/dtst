from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import UpscaleConfig, load_upscale_config
from dtst.embeddings.base import detect_device
from dtst.files import build_save_kwargs, find_images, resolve_dirs
from dtst.sidecar import copy_sidecar, read_sidecar, scale_classes, write_sidecar

logger = logging.getLogger(__name__)

MODEL_PRESETS: dict[int, dict[str, str]] = {
    4: {
        "name": "realesrgan-x4",
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    2: {
        "name": "realesrgan-x2",
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
}

PRESET_BY_NAME: dict[str, dict[str, str]] = {
    p["name"]: p for p in MODEL_PRESETS.values()
}

DENOISE_MODELS = {
    "general": {
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    },
    "wdn": {
        "filename": "realesr-general-wdn-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    },
}

MODELS_DIR = Path.home() / ".cache" / "dtst" / "models"


def _download_model(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    logger.info("Downloading model to %s", dest)
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading model"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _resolve_model_path(model: str | None, scale: int) -> Path:
    if model is None:
        preset = MODEL_PRESETS[scale]
        dest = MODELS_DIR / preset["filename"]
        if not dest.exists():
            _download_model(preset["url"], dest)
        return dest

    path = Path(model)
    if path.exists() and path.is_file():
        return path

    if model in PRESET_BY_NAME:
        preset = PRESET_BY_NAME[model]
        dest = MODELS_DIR / preset["filename"]
        if not dest.exists():
            _download_model(preset["url"], dest)
        return dest

    raise click.ClickException(
        f"Model not found: {model!r}. "
        f"Provide a path to a .pth file or use a preset: {sorted(PRESET_BY_NAME)}"
    )


def _load_denoise_model(
    denoise: float, device: torch.device
) -> tuple[torch.nn.Module, int]:
    import spandrel

    general_info = DENOISE_MODELS["general"]
    wdn_info = DENOISE_MODELS["wdn"]

    general_path = MODELS_DIR / general_info["filename"]
    wdn_path = MODELS_DIR / wdn_info["filename"]

    if not general_path.exists():
        _download_model(general_info["url"], general_path)
    if not wdn_path.exists():
        _download_model(wdn_info["url"], wdn_path)

    logger.info("Loading denoise models (strength=%.2f) on %s", denoise, device)

    raw_a = torch.load(general_path, map_location="cpu", weights_only=True)
    raw_b = torch.load(wdn_path, map_location="cpu", weights_only=True)

    params_a = raw_a.get("params", raw_a)
    params_b = raw_b.get("params", raw_b)

    interpolated = {}
    for k in params_a:
        interpolated[k] = denoise * params_a[k] + (1 - denoise) * params_b[k]

    descriptor = spandrel.ModelLoader(device=device).load_from_state_dict(interpolated)
    descriptor.model.eval()
    return descriptor.model, descriptor.scale


def _tile_upscale(
    model: torch.nn.Module,
    img: torch.Tensor,
    scale: int,
    tile_size: int,
    tile_pad: int,
    device: torch.device,
) -> torch.Tensor:
    if tile_size == 0:
        return model(img.unsqueeze(0).to(device)).squeeze(0).cpu()

    _, h, w = img.shape
    out = torch.empty(3, h * scale, w * scale)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            in_y1 = max(y - tile_pad, 0)
            in_y2 = min(y + tile_size + tile_pad, h)
            in_x1 = max(x - tile_pad, 0)
            in_x2 = min(x + tile_size + tile_pad, w)

            tile = img[:, in_y1:in_y2, in_x1:in_x2]

            with torch.no_grad():
                upscaled = model(tile.unsqueeze(0).to(device)).squeeze(0).cpu()

            pad_top = (y - in_y1) * scale
            pad_left = (x - in_x1) * scale
            out_h = min(tile_size, h - y) * scale
            out_w = min(tile_size, w - x) * scale

            out[:, y * scale : y * scale + out_h, x * scale : x * scale + out_w] = (
                upscaled[:, pad_top : pad_top + out_h, pad_left : pad_left + out_w]
            )

    return out


def _load_and_preprocess(path: Path) -> tuple[Path, torch.Tensor | None, str | None]:
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        img.close()
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return path, tensor, None
    except Exception as e:
        return path, None, str(e)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: list[str] | None,
    to: str | None,
    scale: int | None,
    model: str | None,
    tile_size: int | None,
    tile_pad: int | None,
    fmt: str | None,
    quality: int | None,
    denoise: float | None,
) -> UpscaleConfig:
    if config is not None:
        cfg = load_upscale_config(config)
    else:
        cfg = UpscaleConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = from_dirs
    if to is not None:
        cfg.to = to
    if scale is not None:
        cfg.scale = scale
    if model is not None:
        cfg.model = model
    if tile_size is not None:
        cfg.tile_size = tile_size
    if tile_pad is not None:
        cfg.tile_pad = tile_pad
    if fmt is not None:
        cfg.format = fmt
    if quality is not None:
        cfg.quality = quality
    if denoise is not None:
        cfg.denoise = denoise

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'upscale.from' in config)")
    if cfg.to is None:
        raise click.ClickException("--to is required (or set 'upscale.to' in config)")
    if cfg.denoise is not None and cfg.scale != 4:
        raise click.ClickException("--denoise is only available with 4x upscaling")
    if cfg.denoise is not None and cfg.model is not None:
        raise click.ClickException("--denoise is not compatible with --model")

    return cfg


@click.command("upscale")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs, e.g. 'images/*').")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--scale", "-s", type=click.Choice(["2", "4"]), default=None, help="Upscale factor. Ignored when --model is provided (default: 4).")
@click.option("--model", "-m", type=str, default=None, help="Model preset name or path to a .pth file. Overrides --scale.")
@click.option("--tile-size", "-t", type=int, default=None, help="Tile size in pixels for processing; 0 disables tiling (default: 512).")
@click.option("--tile-pad", type=int, default=None, help="Overlap padding between tiles in pixels (default: 32).")
@click.option("--format", "-f", "fmt", type=click.Choice(["jpg", "png", "webp"]), default=None, help="Output image format. Default preserves the source format.")
@click.option("--quality", "-q", type=int, default=None, help="JPEG/WebP output quality, 1-100 (default: 95).")
@click.option("--denoise", "-n", type=float, default=None, help="Denoise strength 0.0-1.0. Lower preserves more texture. Only available at 4x.")
@click.option("--workers", "-w", type=int, default=None, help="Number of threads for image preloading (default: 4).")
@click.option("--dry-run", is_flag=True, help="Preview what would be written without processing.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    scale: str | None,
    model: str | None,
    tile_size: int | None,
    tile_pad: int | None,
    fmt: str | None,
    quality: int | None,
    denoise: float | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Upscale images using AI super-resolution models.

    Reads images from one or more source folders and writes upscaled
    copies to a destination folder. Uses spandrel to load PyTorch
    super-resolution models (Real-ESRGAN, SwinIR, HAT, etc.).

    By default uses a 4x Real-ESRGAN model. Use --scale to choose
    between 2x and 4x upscaling, or --model to provide a custom
    .pth weights file (scale is auto-detected from the model).

    Use --denoise to control how much denoising is applied (4x only).
    0.0 preserves the most texture, 1.0 applies full denoising.
    This activates a lighter general-purpose model with controllable
    denoise strength via weight interpolation.

    Large images are processed in tiles to avoid GPU memory issues.
    Adjust --tile-size to control memory usage (smaller = less VRAM).

    \b
    Examples:
        dtst upscale -d ./project --from faces --to upscaled
        dtst upscale -d ./project --from faces --to upscaled --scale 2
        dtst upscale -d ./project --from faces --to upscaled --denoise 0.5
        dtst upscale -d ./project --from faces --to upscaled --model ./custom.pth
        dtst upscale config.yaml --dry-run
    """
    parsed_from_dirs: list[str] | None = None
    if from_dirs is not None:
        parsed_from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
        if not parsed_from_dirs:
            raise click.ClickException("--from must contain at least one folder name")

    parsed_scale: int | None = int(scale) if scale is not None else None

    cfg = _resolve_config(
        config, working_dir, parsed_from_dirs, to, parsed_scale, model,
        tile_size, tile_pad, fmt, quality, denoise,
    )

    input_dirs = resolve_dirs(cfg.working_dir, cfg.from_dirs)
    output_dir = cfg.working_dir / cfg.to

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

    from_label = ", ".join(str(d) for d in input_dirs)
    num_workers = workers if workers is not None else 4

    if dry_run:
        if cfg.denoise is not None:
            model_label = f"realesr-general-x4v3 (denoise={cfg.denoise:.2f})"
        else:
            model_path = _resolve_model_path(cfg.model, cfg.scale)
            model_label = model_path.name
        click.echo(f"\nDry run -- would upscale {len(images):,} images")
        click.echo(f"  Model: {model_label}")
        click.echo(f"  Source: {from_label}")
        click.echo(f"  Output: {output_dir}")
        return

    device = torch.device(detect_device())

    if cfg.denoise is not None:
        sr_model, actual_scale = _load_denoise_model(cfg.denoise, device)
        model_label = f"realesr-general-x4v3 (denoise={cfg.denoise:.2f})"
    else:
        import spandrel

        model_path = _resolve_model_path(cfg.model, cfg.scale)
        logger.info("Loading model %s on %s", model_path.name, device)
        model_descriptor = spandrel.ModelLoader(device=device).load_from_file(model_path)
        sr_model = model_descriptor.model
        sr_model.eval()
        actual_scale = model_descriptor.scale
        model_label = model_path.name

        if cfg.model is not None and cfg.model != model_path.name:
            logger.info("Model scale: %dx (auto-detected from weights)", actual_scale)

    logger.info(
        "Upscaling %d images %dx from [%s] (tile=%d, device=%s)",
        len(images), actual_scale, from_label, cfg.tile_size, device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    ok_count = 0
    failed_count = 0

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=num_workers) as loader:
            results_iter = loader.map(_load_and_preprocess, images)

            with tqdm(total=len(images), desc="Upscaling", unit="image") as pbar:
                for img_path, tensor, load_error in results_iter:
                    name = img_path.name

                    if tensor is None:
                        failed_count += 1
                        logger.error("Failed to load %s: %s", name, load_error)
                        pbar.set_postfix(ok=ok_count, fail=failed_count)
                        pbar.update(1)
                        continue

                    try:
                        upscaled_tensor = _tile_upscale(
                            sr_model, tensor, actual_scale,
                            cfg.tile_size, cfg.tile_pad, device,
                        )

                        result_arr = (
                            upscaled_tensor.clamp(0, 1)
                            .permute(1, 2, 0)
                            .mul(255)
                            .byte()
                            .numpy()
                        )
                        result_img = Image.fromarray(result_arr)

                        if cfg.format is not None:
                            out_name = img_path.stem + "." + cfg.format
                        else:
                            out_name = name

                        save_kwargs = build_save_kwargs(Path(out_name), quality=cfg.quality)

                        result_img.save(output_dir / out_name, **save_kwargs)
                        result_img.close()
                        copy_sidecar(img_path, output_dir / out_name, exclude={"metrics"})
                        classes = read_sidecar(img_path).get("classes")
                        upscale_data = {"upscale": actual_scale}
                        if classes:
                            upscale_data["classes"] = scale_classes(classes, actual_scale)
                        write_sidecar(output_dir / out_name, upscale_data)
                        ok_count += 1

                    except torch.cuda.OutOfMemoryError:
                        failed_count += 1
                        logger.error(
                            "GPU out of memory on %s -- try reducing --tile-size", name
                        )
                    except Exception as e:
                        failed_count += 1
                        logger.error("Failed to upscale %s: %s", name, e)

                    pbar.set_postfix(ok=ok_count, fail=failed_count)
                    pbar.update(1)

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nUpscale complete!")
    click.echo(f"  Upscaled: {ok_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Scale: {actual_scale}x ({model_label})")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
