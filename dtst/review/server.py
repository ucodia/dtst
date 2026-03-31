from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from PIL import Image

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from dtst.files import find_images, move_image
from dtst.sidecar import read_sidecar

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).parent / "index.html"


class ApplyRequest(BaseModel):
    view: str
    selected: list[str]


def create_app(source_dir: Path, filtered_dir: Path) -> FastAPI:
    app = FastAPI(title="dtst review")

    def resolve_dir(view: str) -> Path:
        return filtered_dir if view == "filtered" else source_dir

    def image_info(img_path: Path, view: str) -> dict:
        sidecar = read_sidecar(img_path)
        file_size = img_path.stat().st_size
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            width, height = 0, 0
        return {
            "filename": img_path.name,
            "url": f"/images/{view}/{img_path.name}",
            "width": width,
            "height": height,
            "file_size": file_size,
            "sidecar": sidecar or None,
        }

    def move_selected(from_dir: Path, to_dir: Path, filenames: set[str]) -> tuple[int, list[str]]:
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

    @app.get("/api/images")
    async def list_images(view: str = Query("source")):
        target = resolve_dir(view)
        images = [
            image_info(p, view) for p in find_images(target)
        ] if target.is_dir() else []

        return {
            "view": view,
            "images": images,
            "source_count": len(find_images(source_dir)) if source_dir.is_dir() else 0,
            "filtered_count": len(find_images(filtered_dir)) if filtered_dir.is_dir() else 0,
        }

    @app.get("/images/{view}/{filename}")
    async def serve_image(view: str, filename: str):
        directory = resolve_dir(view)
        file_path = directory / filename
        if not file_path.is_file() or not file_path.resolve().is_relative_to(directory.resolve()):
            return JSONResponse({"error": "not found"}, status_code=404)

        content_type, _ = mimetypes.guess_type(str(file_path))
        return FileResponse(
            file_path,
            media_type=content_type or "application/octet-stream",
            headers={"Cache-Control": "public, max-age=60"},
        )

    @app.post("/api/apply")
    async def apply_filter(req: ApplyRequest):
        selected_set = set(req.selected)

        if req.view == "source":
            # Move unselected images from source to filtered
            all_names = {p.name for p in find_images(source_dir)} if source_dir.is_dir() else set()
            to_move = all_names - selected_set
            moved, errors = move_selected(source_dir, filtered_dir, to_move)
        elif req.view == "filtered":
            # Move selected images from filtered back to source
            moved, errors = move_selected(filtered_dir, source_dir, selected_set)
            try:
                filtered_dir.rmdir()
            except OSError:
                pass
        else:
            moved, errors = 0, [f"Unknown view: {req.view}"]

        logger.info("Applied: moved %d, errors %d", moved, len(errors))
        return {"moved": moved, "errors": errors}

    return app
