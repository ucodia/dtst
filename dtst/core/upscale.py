"""Library-layer implementation of ``dtst upscale``."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.embeddings.base import detect_device
from dtst.errors import InputError
from dtst.files import build_save_kwargs, find_images, resolve_dirs
from dtst.results import UpscaleResult
from dtst.sidecar import (
    EXCLUDE_METRICS,
    copy_sidecar,
    read_sidecar,
    scale_classes,
    write_sidecar,
)

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


def _download_model(url: str, dest: Path, progress: bool = True) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    logger.info("Downloading model to %s", dest)
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(tmp, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc="Downloading model",
                disable=not progress,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _resolve_model_path(model: str | None, scale: int, progress: bool = True) -> Path:
    if model is None:
        preset = MODEL_PRESETS[scale]
        dest = MODELS_DIR / preset["filename"]
        if not dest.exists():
            _download_model(preset["url"], dest, progress=progress)
        return dest

    path = Path(model)
    if path.exists() and path.is_file():
        return path

    if model in PRESET_BY_NAME:
        preset = PRESET_BY_NAME[model]
        dest = MODELS_DIR / preset["filename"]
        if not dest.exists():
            _download_model(preset["url"], dest, progress=progress)
        return dest

    raise InputError(
        f"Model not found: {model!r}. "
        f"Provide a path to a .pth file or use a preset: {sorted(PRESET_BY_NAME)}"
    )


def _load_denoise_model(
    denoise: float, device: torch.device, progress: bool = True
) -> tuple[torch.nn.Module, int]:
    import spandrel

    general_info = DENOISE_MODELS["general"]
    wdn_info = DENOISE_MODELS["wdn"]

    general_path = MODELS_DIR / general_info["filename"]
    wdn_path = MODELS_DIR / wdn_info["filename"]

    if not general_path.exists():
        _download_model(general_info["url"], general_path, progress=progress)
    if not wdn_path.exists():
        _download_model(wdn_info["url"], wdn_path, progress=progress)

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


def upscale(
    *,
    from_dirs: str,
    to: str,
    scale: int = 4,
    model: str | None = None,
    tile_size: int = 512,
    tile_pad: int = 32,
    fmt: str | None = None,
    quality: int = 95,
    denoise: float | None = None,
    workers: int = 4,
    dry_run: bool = False,
    progress: bool = True,
) -> UpscaleResult:
    """Upscale images using AI super-resolution models."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")

    if denoise is not None and scale != 4:
        raise InputError("denoise is only available with 4x upscaling")
    if denoise is not None and model is not None:
        raise InputError("denoise is not compatible with model")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    input_dirs = resolve_dirs(dirs_list)
    output_dir = Path(to).expanduser().resolve()

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise InputError(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: "
            f"{', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise InputError(f"No images found in: {', '.join(str(d) for d in input_dirs)}")

    from_label = ", ".join(str(d) for d in input_dirs)

    if dry_run:
        if denoise is not None:
            model_label = f"realesr-general-x4v3 (denoise={denoise:.2f})"
        else:
            model_path = _resolve_model_path(model, scale, progress=progress)
            model_label = model_path.name
        return UpscaleResult(
            ok=0,
            failed=0,
            scale=scale,
            model_label=model_label,
            output_dir=output_dir,
            dry_run=True,
            total_images=len(images),
            from_label=from_label,
            elapsed=0.0,
        )

    device = torch.device(detect_device())

    if denoise is not None:
        sr_model, actual_scale = _load_denoise_model(denoise, device, progress=progress)
        model_label = f"realesr-general-x4v3 (denoise={denoise:.2f})"
    else:
        import spandrel

        model_path = _resolve_model_path(model, scale, progress=progress)
        logger.info("Loading model %s on %s", model_path.name, device)
        model_descriptor = spandrel.ModelLoader(device=device).load_from_file(
            model_path
        )
        sr_model = model_descriptor.model
        sr_model.eval()
        actual_scale = model_descriptor.scale
        model_label = model_path.name

        if model is not None and model != model_path.name:
            logger.info("Model scale: %dx (auto-detected from weights)", actual_scale)

    logger.info(
        "Upscaling %d images %dx from [%s] (tile=%d, device=%s)",
        len(images),
        actual_scale,
        from_label,
        tile_size,
        device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    ok_count = 0
    failed_count = 0

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=workers) as loader:
            results_iter = loader.map(_load_and_preprocess, images)

            with tqdm(
                total=len(images),
                desc="Upscaling",
                unit="image",
                disable=not progress,
            ) as pbar:
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
                            sr_model,
                            tensor,
                            actual_scale,
                            tile_size,
                            tile_pad,
                            device,
                        )

                        result_arr = (
                            upscaled_tensor.clamp(0, 1)
                            .permute(1, 2, 0)
                            .mul(255)
                            .byte()
                            .numpy()
                        )
                        result_img = Image.fromarray(result_arr)

                        out_name = (
                            img_path.stem + "." + fmt if fmt is not None else name
                        )

                        save_kwargs = build_save_kwargs(Path(out_name), quality=quality)

                        result_img.save(output_dir / out_name, **save_kwargs)
                        result_img.close()
                        copy_sidecar(
                            img_path, output_dir / out_name, exclude=EXCLUDE_METRICS
                        )
                        classes = read_sidecar(img_path).get("classes")
                        upscale_data = {"upscale": actual_scale}
                        if classes:
                            upscale_data["classes"] = scale_classes(
                                classes, actual_scale
                            )
                        write_sidecar(output_dir / out_name, upscale_data)
                        ok_count += 1

                    except torch.cuda.OutOfMemoryError:
                        failed_count += 1
                        logger.error(
                            "GPU out of memory on %s -- try reducing tile_size", name
                        )
                    except Exception as e:
                        failed_count += 1
                        logger.error("Failed to upscale %s: %s", name, e)

                    pbar.set_postfix(ok=ok_count, fail=failed_count)
                    pbar.update(1)

    return UpscaleResult(
        ok=ok_count,
        failed=failed_count,
        scale=actual_scale,
        model_label=model_label,
        output_dir=output_dir,
        dry_run=False,
        total_images=len(images),
        from_label=from_label,
        elapsed=time.monotonic() - start_time,
    )
