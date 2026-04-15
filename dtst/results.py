"""Result dataclasses returned by :mod:`dtst.core` functions.

Each command's core function returns one of these so library callers
get structured data instead of parsing CLI output.  The CLI layer
formats them for human display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RenameResult:
    renamed: int
    dry_run: bool
    elapsed: float


@dataclass
class ValidateResult:
    total: int
    dim_counts: dict[tuple[int, int], int]
    mode_counts: dict[str, int]
    non_square: int
    total_png: int
    compressed_png: int
    failed: int
    square_checked: bool
    elapsed: float

    @property
    def passed(self) -> bool:
        dims_ok = len(self.dim_counts) <= 1
        modes_ok = len(self.mode_counts) <= 1
        square_ok = (not self.square_checked) or self.non_square == 0
        errors_ok = self.failed == 0
        return dims_ok and modes_ok and square_ok and errors_ok


@dataclass
class ClusterInfo:
    rank: int
    label: int
    size: int


@dataclass
class ClusterResult:
    model: str
    total_images: int
    embedded_images: int
    clusters: list[ClusterInfo] = field(default_factory=list)
    noise_images: int = 0
    output_dir: Path | None = None
    elapsed: float = 0.0


@dataclass
class AnalyzeResult:
    analyzed: int
    skipped: int
    failed: int
    cleared: int
    dry_run: bool
    elapsed: float


@dataclass
class AnnotateResult:
    annotated: int
    skipped: int
    dry_run: bool
    elapsed: float


@dataclass
class AugmentResult:
    ok: int
    failed: int
    files_written: int
    transforms: list[str]
    copy_originals: bool
    total_output_estimate: int
    output_dir: Path
    dry_run: bool
    elapsed: float


@dataclass
class DedupResult:
    mode: str  # "dedup" | "restore" | "noop"
    groups: int = 0
    kept: int = 0
    moved: int = 0
    restored: int = 0
    errors: int = 0
    dry_run: bool = False
    source_dir: Path | None = None
    duplicated_dir: Path | None = None
    losers_preview: list[tuple[str, str]] = field(default_factory=list)
    total_losers: int = 0
    elapsed: float = 0.0
    message: str | None = None


@dataclass
class DetectResult:
    processed: int
    failed: int
    class_counts: dict[str, int]
    valid: int
    cleared: int
    dry_run: bool
    elapsed: float


@dataclass
class ExtractClassesResult:
    processed: int
    crops_extracted: int
    no_detections: int
    failed: int
    output_dir: Path
    dry_run: bool
    dry_run_dets: int = 0
    elapsed: float = 0.0


@dataclass
class ExtractFacesResult:
    processed: int
    faces_extracted: int
    no_faces: int
    failed: int
    output_dir: Path
    elapsed: float


@dataclass
class ExtractFramesResult:
    processed: int
    frames_extracted: int
    skipped: int
    failed: int
    output_dir: Path
    dry_run: bool
    total_videos: int
    keyframes: float
    fmt: str
    elapsed: float


@dataclass
class FetchResult:
    downloaded: int
    skipped_existing: int
    skipped_unsupported: int
    failed: int
    rate_limited: int
    rate_limited_domains: list[str]
    output_dir: Path
    elapsed: float


@dataclass
class FormatResult:
    converted: int
    failed: int
    output_dir: Path
    dry_run: bool
    fmt: str | None
    channels: str | None
    strip_metadata: bool
    quality: int
    compress_level: int
    total_images: int
    elapsed: float


@dataclass
class FrameResult:
    resized: int
    failed: int
    output_dir: Path
    dry_run: bool
    width: int | None
    height: int | None
    mode: str
    gravity: str
    fill: str
    fill_color: str
    total_images: int
    elapsed: float


@dataclass
class SearchResult:
    queries_run: int
    engines: list[str]
    engine_counts: dict[str, int]
    total_unique: int
    new_urls: int
    errors: int
    output_file: Path
    dry_run: bool
    queries_preview: list[str] = field(default_factory=list)
    taxon_ids: list[int] = field(default_factory=list)
    min_size: int = 0
    elapsed: float = 0.0


@dataclass
class SelectResult:
    ok: int
    skipped: int
    excluded: int
    failed: int
    total_images: int
    selected: int
    move: bool
    output_dir: Path
    from_label: str
    dry_run: bool
    rejects_preview: list[tuple[str, str]] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class UpscaleResult:
    ok: int
    failed: int
    scale: int
    model_label: str
    output_dir: Path
    dry_run: bool
    total_images: int
    from_label: str
    elapsed: float
