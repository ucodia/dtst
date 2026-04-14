from __future__ import annotations

import logging
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

from PIL import Image

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from dtst.files import IMAGE_EXTENSIONS, find_images, move_image
from dtst.sidecar import read_sidecar

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).parent / "index.html"


class ApplyRequest(BaseModel):
    view: str
    selected: list[str]


class SelectRequest(BaseModel):
    from_dir: str
    to: str


def create_app(
    working_dir: Path,
    source_dir: Path | None = None,
    filtered_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="dtst review")

    state: dict[str, Path | None] = {
        "source_dir": source_dir,
        "filtered_dir": filtered_dir,
    }

    dim_cache: dict[tuple[str, int, int], tuple[int, int]] = {}
    DIM_CACHE_MAX = 50000

    def resolve_dir(view: str) -> Path | None:
        if view == "filtered":
            return state["filtered_dir"]
        return state["source_dir"]

    def image_info(img_path: Path, view: str) -> dict:
        sidecar = read_sidecar(img_path)
        try:
            st = img_path.stat()
            file_size = st.st_size
            cache_key = (str(img_path), st.st_mtime_ns, file_size)
        except OSError:
            file_size = 0
            cache_key = None

        dims = dim_cache.get(cache_key) if cache_key is not None else None
        if dims is None:
            try:
                with Image.open(img_path) as im:
                    dims = im.size
            except Exception:
                dims = (0, 0)
            if cache_key is not None and len(dim_cache) < DIM_CACHE_MAX:
                dim_cache[cache_key] = dims
        width, height = dims
        return {
            "filename": img_path.name,
            "url": f"/images/{view}/{img_path.name}",
            "width": width,
            "height": height,
            "file_size": file_size,
            "sidecar": sidecar or None,
        }

    def move_selected(
        from_dir: Path, to_dir: Path, filenames: set[str]
    ) -> tuple[int, list[str]]:
        """Move images matching *filenames* from one directory to another.

        Returns (moved_count, error_list).
        """
        if not from_dir.is_dir():
            return 0, [f"Directory does not exist: {from_dir}"]
        to_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        errors: list[str] = []
        for img_path in find_images(from_dir):
            if img_path.name not in filenames:
                continue
            dest = to_dir / img_path.name
            if dest.exists():
                errors.append(f"{img_path.name}: already exists in destination")
                continue
            try:
                move_image(img_path, dest)
                moved += 1
            except OSError as e:
                errors.append(f"{img_path.name}: {e}")
        return moved, errors

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _HTML_PATH.read_text()

    @app.get("/api/config")
    async def get_config():
        src = state["source_dir"]
        flt = state["filtered_dir"]
        if src is not None and flt is not None:
            return {
                "configured": True,
                "from_dir": str(src.relative_to(working_dir)),
                "to": flt.name,
            }
        return {"configured": False, "from_dir": None, "to": None}

    @app.get("/api/buckets")
    async def list_buckets():
        buckets = []
        for dirpath, dirnames, filenames in os.walk(working_dir, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            has_img = any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS for f in filenames
            )
            if not has_img:
                continue
            dp = Path(dirpath)
            rel = str(dp.relative_to(working_dir))
            subdirs = [d for d in dirnames]
            buckets.append({"path": rel, "subdirs": subdirs})
        return {"buckets": buckets}

    @app.post("/api/select")
    async def select_buckets(req: SelectRequest):
        target = working_dir / req.from_dir
        if not target.is_dir():
            return JSONResponse(
                {"error": f"Directory does not exist: {req.from_dir}"},
                status_code=400,
            )
        state["source_dir"] = target
        state["filtered_dir"] = target / req.to
        logger.info("Buckets selected: from=%s, to=%s", req.from_dir, req.to)
        return {"ok": True}

    @app.get("/api/images")
    async def list_images(view: str = Query("source")):
        src = state["source_dir"]
        flt = state["filtered_dir"]
        if src is None:
            return {
                "configured": False,
                "view": view,
                "images": [],
                "source_count": 0,
                "filtered_count": 0,
            }

        target = resolve_dir(view)
        source_paths = find_images(src) if src.is_dir() else []
        if flt is not None and flt.is_dir():
            filtered_paths = (
                source_paths if target == src and flt == src else find_images(flt)
            )
        else:
            filtered_paths = []

        if target is None or not target.is_dir():
            target_paths = []
        elif target == src:
            target_paths = source_paths
        elif target == flt:
            target_paths = filtered_paths
        else:
            target_paths = find_images(target)

        if target_paths:
            workers = min(32, (cpu_count() or 4) * 4)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                images = list(executor.map(lambda p: image_info(p, view), target_paths))
        else:
            images = []

        return {
            "configured": True,
            "view": view,
            "images": images,
            "source_count": len(source_paths),
            "filtered_count": len(filtered_paths),
        }

    @app.get("/images/{view}/{filename}")
    async def serve_image(view: str, filename: str):
        directory = resolve_dir(view)
        if directory is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        file_path = directory / filename
        if not file_path.is_file() or not file_path.resolve().is_relative_to(
            directory.resolve()
        ):
            return JSONResponse({"error": "not found"}, status_code=404)

        content_type, _ = mimetypes.guess_type(str(file_path))
        return FileResponse(
            file_path,
            media_type=content_type or "application/octet-stream",
            headers={"Cache-Control": "public, max-age=60"},
        )

    @app.post("/api/apply")
    async def apply_filter(req: ApplyRequest):
        src = state["source_dir"]
        flt = state["filtered_dir"]
        if src is None or flt is None:
            return JSONResponse(
                {"error": "No buckets configured"},
                status_code=400,
            )

        selected_set = set(req.selected)

        if req.view == "source":
            # Move unselected images from source to filtered
            all_names = {p.name for p in find_images(src)} if src.is_dir() else set()
            to_move = all_names - selected_set
            moved, errors = move_selected(src, flt, to_move)
        elif req.view == "filtered":
            # Move selected images from filtered back to source
            moved, errors = move_selected(flt, src, selected_set)
            try:
                flt.rmdir()
            except OSError:
                pass
        else:
            moved, errors = 0, [f"Unknown view: {req.view}"]

        logger.info("Applied: moved %d, errors %d", moved, len(errors))
        return {"moved": moved, "errors": errors}

    return app
