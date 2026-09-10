"""YOLO MVP oil-candidate detector.

This module implements a separate detection pathway that uses an Ultralytics YOLO
object-detection model to find oil-candidate bounding boxes in Sentinel-1 SAR
imagery, then derives approximate polygon contours from the SAR data within each
box.

Architecture
------------
The YOLO detector is intentionally kept **separate** from the ONNX segmentation
pipeline.  It does not share the segmentation model's ImageNet-normalised tensor;
instead it renders its own 8-bit RGB image from the filtered dB SAR data using a
configurable dB window.

.. warning::

   **SAR-to-YOLO rendering is an unvalidated assumption.** The checkpoint was
   trained on the ``mmuthukumar07/oil-spill-dataset`` whose original SAR-to-image
   preprocessing is undocumented.  The dB window used here is a best-effort match
   and MUST be validated against a sample of the original training images before
   operational use.  See :data:`DEFAULT_YOLO_DB_WINDOW`.

The derived contour inside each YOLO bounding box is **image-derived and
approximate** — it is NOT the output of a semantic-segmentation model.  The output
contract labels this transparently via ``geometry_source`` and ``geometry_quality``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oilspill.detectors.contracts import (
    CandidateResult,
    ConfidenceAdjustment,
    DetectionOutput,
    DetectorType,
    GeometryQuality,
    GeometrySource,
)

if TYPE_CHECKING:
    from affine import Affine
    from rasterio.crs import CRS
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: dB window for the SAR-to-YOLO renderer.  See module docstring.
DEFAULT_YOLO_DB_WINDOW: tuple[float, float] = (-25.0, 0.0)

#: Default YOLO inference confidence threshold.
DEFAULT_CONF_THRESHOLD: float = 0.25

#: Default IoU threshold for non-maximum suppression.
DEFAULT_IOU_THRESHOLD: float = 0.45

#: Default tile size matching the YOLO checkpoint's ``imgsz``.
DEFAULT_TILE_SIZE: int = 1024

#: Default tile overlap in pixels.
DEFAULT_TILE_OVERLAP: int = 128

#: Minimum connected-component size (pixels) to retain during contour extraction.
DEFAULT_CONTOUR_MIN_PIXELS: int = 50

#: Structuring-element side length for morphological opening/closing.
DEFAULT_MORPH_SIZE: int = 3

#: Polygon simplification tolerance (in projected CRS units, e.g. metres).
DEFAULT_SIMPLIFY_TOLERANCE: float = 1.0

#: Wind speed threshold (m/s) below which a low-wind look-alike flag is raised.
DEFAULT_LOW_WIND_THRESHOLD: float = 3.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class YoloDetectorConfig:
    """All tunables for the YOLO MVP detector."""

    weights_path: Path | None = None
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    db_min: float = DEFAULT_YOLO_DB_WINDOW[0]
    db_max: float = DEFAULT_YOLO_DB_WINDOW[1]
    contour_min_pixels: int = DEFAULT_CONTOUR_MIN_PIXELS
    morph_size: int = DEFAULT_MORPH_SIZE
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE
    low_wind_threshold: float = DEFAULT_LOW_WIND_THRESHOLD


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class TileInfo:
    """Pixel coordinates and index of one tile."""

    index: int
    y: int
    x: int
    h: int
    w: int


@dataclass
class RawDetection:
    """A single YOLO detection mapped to scene pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    tile_index: int


@dataclass
class ContourResult:
    """Result of contour extraction within a bounding box."""

    mask: NDArray[np.uint8] | None
    """Binary mask of the selected component (bbox-relative), or ``None``."""

    success: bool
    """Whether a stable contour was found."""

    quality_flags: list[str] = field(default_factory=list)
    """Flags raised during extraction (e.g. ``'contour_extraction_failed'``)."""

    component_darkness: float = 0.0
    """Mean dB value of the selected component (lower = darker = more oil-like)."""

    component_area_px: int = 0
    """Area of the selected component in pixels."""


# ---------------------------------------------------------------------------
# SAR-to-YOLO image renderer
# ---------------------------------------------------------------------------


def sar_to_yolo_image(
    db_image: NDArray[np.floating[Any]],
    db_min: float = DEFAULT_YOLO_DB_WINDOW[0],
    db_max: float = DEFAULT_YOLO_DB_WINDOW[1],
) -> NDArray[np.uint8]:
    """Render a filtered dB SAR image to an 8-bit 3-channel image for YOLO.

    The mapping is a linear window: ``pixel = clip((db - db_min)/(db_max - db_min), 0, 1) * 255``,
    replicated to three channels.

    .. warning::

       This mapping is an **explicit configurable assumption**.  The original
       training images' dB-to-byte conversion is undocumented.  Validate against
       a sample of training images before operational use.  See the module
       docstring and ``docs/yolo_mvp.md``.

    Parameters
    ----------
    db_image:
        Backscatter in dB, shape ``(H, W)``.
    db_min, db_max:
        dB window mapped to 0 and 255 respectively.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 array.
    """
    if db_min >= db_max:
        raise ValueError(f"require db_min < db_max, got db_min={db_min}, db_max={db_max}")
    arr = np.asarray(db_image, dtype=np.float64)
    scaled = (arr - db_min) / (db_max - db_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------


def generate_tiles(
    height: int,
    width: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> list[TileInfo]:
    """Generate overlapping tile coordinates covering ``(height, width)``.

    Every pixel in the scene is covered by at least one tile.  Tiles at the
    right/bottom edges are clamped to the image bounds.

    Parameters
    ----------
    height, width:
        Scene dimensions in pixels.
    tile_size:
        Side length of each square tile.
    overlap:
        Overlap between adjacent tiles in pixels.

    Returns
    -------
    list[TileInfo]
        One entry per tile with scene-relative ``(y, x, h, w)``.
    """
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"require 0 <= overlap < tile_size, got overlap={overlap}")

    stride = tile_size - overlap
    tiles: list[TileInfo] = []
    idx = 0
    y = 0
    while y < height:
        x = 0
        while x < width:
            th = min(tile_size, height - y)
            tw = min(tile_size, width - x)
            tiles.append(TileInfo(index=idx, y=y, x=x, h=th, w=tw))
            idx += 1
            if x + tile_size >= width:
                break
            x += stride
        if y + tile_size >= height:
            break
        y += stride
    return tiles


def _iou(a: RawDetection, b: RawDetection) -> float:
    """Compute intersection-over-union of two axis-aligned boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def global_nms(
    detections: list[RawDetection],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[RawDetection]:
    """Class-aware greedy non-maximum suppression on scene-coordinate detections.

    For each class, detections are sorted by descending confidence and lower-
    confidence duplicates whose IoU with a retained detection exceeds
    ``iou_threshold`` are suppressed.

    Parameters
    ----------
    detections:
        All detections mapped to scene coordinates (possibly from multiple tiles).
    iou_threshold:
        IoU above which a lower-confidence detection is suppressed.

    Returns
    -------
    list[RawDetection]
        Retained detections after NMS.
    """
    if not detections:
        return []

    # Group by class.
    by_class: dict[int, list[RawDetection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    kept: list[RawDetection] = []
    for _cls, dets in by_class.items():
        dets.sort(key=lambda d: d.confidence, reverse=True)
        retained: list[RawDetection] = []
        for det in dets:
            if all(_iou(det, r) < iou_threshold for r in retained):
                retained.append(det)
        kept.extend(retained)

    return kept


def run_tiled_yolo(
    model: Any,
    image: NDArray[np.uint8],
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
) -> list[RawDetection]:
    """Run YOLO inference on overlapping tiles and merge with global NMS.

    Parameters
    ----------
    model:
        An ``ultralytics.YOLO`` model instance.
    image:
        8-bit ``(H, W, 3)`` image.
    tile_size, tile_overlap:
        Tiling parameters.
    conf:
        YOLO confidence threshold.
    iou:
        IoU threshold for global NMS.

    Returns
    -------
    list[RawDetection]
        Merged detections in scene pixel coordinates.
    """
    h, w = image.shape[:2]
    tiles = generate_tiles(h, w, tile_size=tile_size, overlap=tile_overlap)

    all_dets: list[RawDetection] = []
    for tile in tiles:
        crop = image[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w]
        results = model.predict(crop, imgsz=tile_size, conf=conf, verbose=False)
        if not results:
            continue
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box_idx in range(len(boxes)):
                xyxy = boxes.xyxy[box_idx].cpu().numpy().astype(int)
                box_conf = float(boxes.conf[box_idx].cpu().numpy())
                box_cls = int(boxes.cls[box_idx].cpu().numpy())
                all_dets.append(
                    RawDetection(
                        x1=tile.x + int(xyxy[0]),
                        y1=tile.y + int(xyxy[1]),
                        x2=tile.x + int(xyxy[2]),
                        y2=tile.y + int(xyxy[3]),
                        confidence=box_conf,
                        class_id=box_cls,
                        tile_index=tile.index,
                    )
                )

    return global_nms(all_dets, iou_threshold=iou)


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------


def extract_contour(
    sar_db: NDArray[np.floating[Any]],
    bbox: tuple[int, int, int, int],
    *,
    min_pixels: int = DEFAULT_CONTOUR_MIN_PIXELS,
    morph_size: int = DEFAULT_MORPH_SIZE,
) -> ContourResult:
    """Extract a dark-candidate contour from the SAR dB data within a bbox.

    Steps:
    1. Crop the dB SAR data to ``(y1, x1, y2, x2)``.
    2. Apply Otsu thresholding combined with a local quantile guard.
    3. Morphological opening + closing to reduce speckle.
    4. Connected-component analysis.
    5. Score each component on darkness, area, proximity to centre, compactness.
    6. Select the best component.

    Parameters
    ----------
    sar_db:
        Filtered dB SAR image, shape ``(H, W)``.
    bbox:
        ``(x1, y1, x2, y2)`` in pixel coordinates.
    min_pixels:
        Minimum component size to retain.
    morph_size:
        Structuring element side length.

    Returns
    -------
    ContourResult
    """
    import cv2

    x1, y1, x2, y2 = bbox
    crop = sar_db[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return ContourResult(mask=None, success=False, quality_flags=["empty_bbox_crop"])

    # Normalise crop to 0–255 for Otsu.
    cmin, cmax = float(crop.min()), float(crop.max())
    if cmax - cmin < 1e-6:
        return ContourResult(mask=None, success=False, quality_flags=["uniform_sar_crop"])
    norm = ((crop - cmin) / (cmax - cmin) * 255).astype(np.uint8)

    # Otsu threshold — selects dark regions (oil is darker in SAR).
    thresh_val, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Local quantile guard: also require pixel to be below the 30th percentile
    # of the crop to avoid false triggers on uniformly noisy crops.
    q30 = float(np.percentile(crop, 30))
    quantile_mask = (crop <= q30).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, quantile_mask)

    # Morphological opening (remove small bright noise) then closing (fill holes).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Connected components.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    if num_labels <= 1:
        return ContourResult(
            mask=None, success=False, quality_flags=["contour_extraction_failed"]
        )

    crop_h, crop_w = crop.shape
    crop_cx, crop_cy = crop_w / 2.0, crop_h / 2.0
    crop_area = crop_h * crop_w

    best_score = -1.0
    best_label = -1
    best_darkness = 0.0
    best_area = 0

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue

        component_mask = (labels == label_id)
        darkness = float(crop[component_mask].mean())  # dB — lower = darker

        # Normalised darkness score (relative to crop range).
        darkness_score = 1.0 - (darkness - cmin) / (cmax - cmin) if cmax > cmin else 0.5

        # Area score: favour components that fill 5–80% of the bbox.
        area_frac = area / crop_area
        if area_frac < 0.01:
            area_score = 0.1
        elif area_frac > 0.9:
            area_score = 0.3
        else:
            area_score = min(area_frac / 0.3, 1.0)

        # Proximity to bbox centre.
        cx, cy = float(centroids[label_id, 0]), float(centroids[label_id, 1])
        max_dist = math.sqrt(crop_cx**2 + crop_cy**2) or 1.0
        dist = math.sqrt((cx - crop_cx) ** 2 + (cy - crop_cy) ** 2)
        proximity_score = 1.0 - min(dist / max_dist, 1.0)

        # Compactness (circularity).
        contours, _ = cv2.findContours(
            component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            compactness = (4.0 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        else:
            compactness = 0.0
        compactness_score = min(compactness / 0.5, 1.0)  # cap at 1

        # Weighted aggregate score.
        score = (
            0.35 * darkness_score
            + 0.25 * area_score
            + 0.25 * proximity_score
            + 0.15 * compactness_score
        )

        if score > best_score:
            best_score = score
            best_label = label_id
            best_darkness = darkness
            best_area = area

    if best_label < 0:
        return ContourResult(
            mask=None, success=False, quality_flags=["contour_extraction_failed"]
        )

    result_mask = (labels == best_label).astype(np.uint8)
    flags: list[str] = []

    # Flag implausible shapes.
    area_frac = best_area / crop_area
    if area_frac > 0.85:
        flags.append("contour_fills_most_of_bbox")
    if area_frac < 0.02:
        flags.append("contour_very_small")

    return ContourResult(
        mask=result_mask,
        success=True,
        quality_flags=flags,
        component_darkness=best_darkness,
        component_area_px=best_area,
    )


# ---------------------------------------------------------------------------
# Polygonisation and geometry metrics
# ---------------------------------------------------------------------------


def polygonize_contour(
    mask: NDArray[np.uint8],
    bbox_offset: tuple[int, int],
    transform: Affine,
    crs: CRS,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
) -> BaseGeometry | None:
    """Polygonize a contour mask to a georeferenced shapely geometry.

    Parameters
    ----------
    mask:
        Binary ``(H, W)`` mask of the contour (bbox-relative coordinates).
    bbox_offset:
        ``(x_offset, y_offset)`` of the bbox top-left in scene pixels.
    transform:
        Scene affine transform (pixel -> CRS).
    crs:
        Scene CRS.
    simplify_tolerance:
        Tolerance for polygon simplification (CRS units).

    Returns
    -------
    shapely geometry or None.
    """
    from rasterio.features import shapes
    from shapely.geometry import shape

    # Offset the transform to the bbox origin.
    from affine import Affine as AffineClass

    x_off, y_off = bbox_offset
    local_transform = transform * AffineClass.translation(x_off, y_off)

    polys = []
    for geom_dict, value in shapes(mask.astype(np.uint8), transform=local_transform):
        if value == 1:
            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            polys.append(poly)

    if not polys:
        return None

    from shapely.ops import unary_union

    merged = unary_union(polys)
    if merged.is_empty:
        return None

    # Simplify conservatively.
    if simplify_tolerance > 0:
        merged = merged.simplify(simplify_tolerance, preserve_topology=True)

    # Ensure valid.
    if not merged.is_valid:
        merged = merged.buffer(0)

    return merged if not merged.is_empty else None


def bbox_to_polygon(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    transform: Affine,
) -> dict[str, Any]:
    """Convert a pixel-coordinate bounding box to a GeoJSON polygon.

    Parameters
    ----------
    x1, y1, x2, y2:
        Pixel coordinates of the bounding box.
    transform:
        Scene affine transform.

    Returns
    -------
    dict
        GeoJSON Polygon geometry.
    """
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    geo_corners = [transform * (c[0], c[1]) for c in corners]
    return {
        "type": "Polygon",
        "coordinates": [[(c[0], c[1]) for c in geo_corners]],
    }


def compute_geometry_metrics(
    geom: BaseGeometry,
    crs: CRS,
) -> dict[str, float]:
    """Compute geodesically correct area, perimeter, and shape metrics.

    Reuses :func:`oilspill.pipeline.vectorize.polygon_area_km2` for the area
    calculation, ensuring consistency with the existing segmentation pipeline.

    Parameters
    ----------
    geom:
        A shapely geometry (Polygon or MultiPolygon).
    crs:
        CRS of the geometry.

    Returns
    -------
    dict with keys:
        area_km2, perimeter_km, major_axis_km, minor_axis_km, elongation,
        orientation_degrees, centroid_lon, centroid_lat.
    """
    from pyproj import CRS as PyprojCRS
    from pyproj import Geod

    from oilspill.pipeline.vectorize import polygon_area_km2

    area_km2 = polygon_area_km2(geom, crs)

    # Geodesic perimeter.
    geod = Geod(ellps="WGS84")
    pyproj_crs = PyprojCRS.from_user_input(crs)

    if pyproj_crs.is_geographic:
        _, perimeter_m = geod.geometry_area_perimeter(geom)
        perimeter_km = abs(perimeter_m) / 1e3
    else:
        perimeter_km = float(geom.length) / 1e3

    # Centroid (reproject to WGS84 for lon/lat).
    centroid = geom.centroid
    if not pyproj_crs.is_geographic:
        from pyproj import Transformer

        transformer = Transformer.from_crs(pyproj_crs, PyprojCRS.from_epsg(4326), always_xy=True)
        centroid_lon, centroid_lat = transformer.transform(centroid.x, centroid.y)
    else:
        centroid_lon, centroid_lat = centroid.x, centroid.y

    # Minimum rotated rectangle for major/minor axis and orientation.
    mrr = geom.minimum_rotated_rectangle
    if mrr is not None and not mrr.is_empty:
        coords = list(mrr.exterior.coords)
        edges = []
        for i in range(len(coords) - 1):
            dx = coords[i + 1][0] - coords[i][0]
            dy = coords[i + 1][1] - coords[i][1]
            length = math.sqrt(dx**2 + dy**2)
            edges.append((length, dx, dy))
        edges.sort(key=lambda e: e[0], reverse=True)

        if pyproj_crs.is_geographic and len(edges) >= 2:
            # For geographic CRS, edge lengths in degrees are not meaningful;
            # use geodesic distance.
            from pyproj import Geod as GeodClass

            g = GeodClass(ellps="WGS84")
            major_edge = edges[0]
            minor_edge = edges[1]
            # Compute geodesic distance of major/minor edges.
            p1 = coords[0]
            p2 = coords[1]
            _, _, major_m = g.inv(p1[0], p1[1], p2[0], p2[1])
            p3 = coords[2]
            _, _, minor_m = g.inv(p2[0], p2[1], p3[0], p3[1])
            if major_m < minor_m:
                major_m, minor_m = minor_m, major_m
                major_edge, minor_edge = minor_edge, major_edge
            major_axis_km = abs(major_m) / 1e3
            minor_axis_km = abs(minor_m) / 1e3
        elif len(edges) >= 2:
            major_axis_km = edges[0][0] / 1e3
            minor_axis_km = edges[1][0] / 1e3
        else:
            major_axis_km = 0.0
            minor_axis_km = 0.0

        # Orientation of major axis (degrees from north, 0–180).
        if edges:
            _, dx, dy = edges[0]
            angle_rad = math.atan2(dx, dy)  # atan2(east, north) -> from north
            orientation = math.degrees(angle_rad) % 180
        else:
            orientation = 0.0

        elongation = major_axis_km / minor_axis_km if minor_axis_km > 0 else 1.0
    else:
        major_axis_km = 0.0
        minor_axis_km = 0.0
        elongation = 1.0
        orientation = 0.0

    return {
        "area_km2": area_km2,
        "perimeter_km": perimeter_km,
        "major_axis_km": major_axis_km,
        "minor_axis_km": minor_axis_km,
        "elongation": elongation,
        "orientation_degrees": orientation,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
    }


# ---------------------------------------------------------------------------
# Land-overlap screening
# ---------------------------------------------------------------------------


def screen_land_overlap(
    geom: BaseGeometry,
    land_geometries: list[BaseGeometry],
) -> float:
    """Return the fraction of ``geom`` that overlaps any land polygon.

    Parameters
    ----------
    geom:
        Candidate geometry (in the same CRS as ``land_geometries``).
    land_geometries:
        List of land polygons.

    Returns
    -------
    float
        Overlap fraction in [0, 1].
    """
    if not land_geometries or geom.is_empty:
        return 0.0

    from shapely.ops import unary_union

    land = unary_union(land_geometries)
    intersection = geom.intersection(land)
    if intersection.is_empty:
        return 0.0
    geom_area = geom.area
    return float(intersection.area / geom_area) if geom_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Investigation confidence scorer
# ---------------------------------------------------------------------------


def compute_investigation_confidence(
    raw_confidence: float,
    quality_flags: list[str],
    land_overlap_frac: float = 0.0,
    contour_success: bool = True,
    env_context: dict[str, Any] | None = None,
    *,
    low_wind_threshold: float = DEFAULT_LOW_WIND_THRESHOLD,
) -> tuple[float, list[ConfidenceAdjustment]]:
    """Compute the heuristic investigation-confidence score.

    This is a **heuristic ranking score**, NOT a calibrated probability.  It
    starts from the raw YOLO confidence and applies transparent penalties and
    bonuses.  Every adjustment is documented and returned as a breakdown.

    Parameters
    ----------
    raw_confidence:
        Raw YOLO detector confidence (0–1).
    quality_flags:
        Quality flags from contour extraction and other checks.
    land_overlap_frac:
        Fraction of the candidate overlapping land (0–1).
    contour_success:
        Whether contour extraction succeeded.
    env_context:
        Optional dict with keys ``wind_speed_ms`` and/or
        ``optical_corroboration``.
    low_wind_threshold:
        Wind speed threshold for the low-wind look-alike flag.

    Returns
    -------
    tuple[float, list[ConfidenceAdjustment]]
        ``(score, breakdown)`` where ``score`` is in [0, 1].
    """
    score = raw_confidence
    breakdown: list[ConfidenceAdjustment] = [
        ConfidenceAdjustment(reason="Raw model confidence (base)", delta=0.0),
    ]

    # Land overlap penalty.
    if land_overlap_frac > 0.5:
        penalty = -0.4
        score += penalty
        breakdown.append(
            ConfidenceAdjustment(
                reason=f"Land overlap {land_overlap_frac:.0%} (>50%): strong penalty",
                delta=penalty,
            )
        )
    elif land_overlap_frac > 0.1:
        penalty = -0.15
        score += penalty
        breakdown.append(
            ConfidenceAdjustment(
                reason=f"Land overlap {land_overlap_frac:.0%} (>10%): moderate penalty",
                delta=penalty,
            )
        )

    # Contour quality.
    if not contour_success:
        penalty = -0.1
        score += penalty
        breakdown.append(
            ConfidenceAdjustment(
                reason="Contour extraction failed: bbox fallback",
                delta=penalty,
            )
        )

    if "contour_fills_most_of_bbox" in quality_flags:
        penalty = -0.05
        score += penalty
        breakdown.append(
            ConfidenceAdjustment(
                reason="Contour fills >85% of bbox: possibly unstable threshold",
                delta=penalty,
            )
        )

    if "contour_very_small" in quality_flags:
        penalty = -0.05
        score += penalty
        breakdown.append(
            ConfidenceAdjustment(
                reason="Contour very small (<2% of bbox): may be noise",
                delta=penalty,
            )
        )

    # Environmental context.
    ctx = env_context or {}
    wind_speed = ctx.get("wind_speed_ms")
    optical = ctx.get("optical_corroboration")

    if wind_speed is not None:
        if wind_speed < low_wind_threshold:
            penalty = -0.1
            score += penalty
            breakdown.append(
                ConfidenceAdjustment(
                    reason=f"Low wind speed ({wind_speed:.1f} m/s < {low_wind_threshold}): possible look-alike",
                    delta=penalty,
                )
            )
        elif wind_speed > 10.0:
            bonus = 0.05
            score += bonus
            breakdown.append(
                ConfidenceAdjustment(
                    reason=f"Moderate+ wind ({wind_speed:.1f} m/s): dark patch less likely a calm zone",
                    delta=bonus,
                )
            )
    else:
        breakdown.append(
            ConfidenceAdjustment(reason="Wind data unavailable", delta=0.0)
        )

    if optical is True:
        bonus = 0.1
        score += bonus
        breakdown.append(
            ConfidenceAdjustment(
                reason="Optical corroboration available: bonus",
                delta=bonus,
            )
        )
    elif optical is None or optical is False:
        breakdown.append(
            ConfidenceAdjustment(reason="Optical confirmation unavailable", delta=0.0)
        )

    return max(0.0, min(1.0, score)), breakdown


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class YoloDetector:
    """Orchestrates the full YOLO MVP detection pipeline.

    The detector is lazy: YOLO weights are loaded on first use.  If weights
    are not configured, :meth:`detect` raises :class:`FileNotFoundError` with
    a descriptive message (the API translates this to a 503).

    Parameters
    ----------
    config:
        All tunables for the detector.
    """

    def __init__(self, config: YoloDetectorConfig) -> None:
        self.config = config
        self._model: Any = None

    @property
    def available(self) -> bool:
        """Whether YOLO weights are configured and the file exists."""
        return (
            self.config.weights_path is not None
            and self.config.weights_path.exists()
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.config.weights_path is None:
            raise FileNotFoundError(
                "YOLO detector unavailable: OILSPILL_API_YOLO_WEIGHTS is not configured."
            )
        path = self.config.weights_path
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO detector unavailable: weights not found at configured path. "
                f"Set OILSPILL_API_YOLO_WEIGHTS to a valid checkpoint."
            )
        from ultralytics import YOLO

        self._model = YOLO(str(path))
        logger.info("Loaded YOLO weights from %s", path)
        return self._model

    def detect(
        self,
        sar_db: NDArray[np.floating[Any]],
        transform: Affine,
        crs: CRS,
        *,
        scene_id: str = "",
        land_geometries: list[BaseGeometry] | None = None,
        env_context: dict[str, Any] | None = None,
    ) -> DetectionOutput:
        """Run the full YOLO MVP detection pipeline on a filtered dB SAR image.

        Parameters
        ----------
        sar_db:
            Filtered dB SAR image, shape ``(H, W)``.
        transform:
            Scene affine transform (pixel -> CRS).
        crs:
            Scene CRS.
        scene_id:
            Identifier for the scene.
        land_geometries:
            Optional list of land polygons (in scene CRS) for overlap screening.
        env_context:
            Optional environmental context dict.

        Returns
        -------
        DetectionOutput
        """
        model = self._load_model()
        cfg = self.config
        model_id = str(cfg.weights_path.name) if cfg.weights_path else "yolo_mvp"

        # 1. Render SAR -> YOLO image.
        yolo_image = sar_to_yolo_image(sar_db, db_min=cfg.db_min, db_max=cfg.db_max)

        # 2. Tiled YOLO inference + global NMS.
        detections = run_tiled_yolo(
            model,
            yolo_image,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
        )

        # 3. Process each detection.
        candidates: list[CandidateResult] = []
        for det in detections:
            bbox = (det.x1, det.y1, det.x2, det.y2)
            flags: list[str] = []

            # Contour extraction.
            contour = extract_contour(
                sar_db,
                bbox,
                min_pixels=cfg.contour_min_pixels,
                morph_size=cfg.morph_size,
            )
            flags.extend(contour.quality_flags)

            # Polygonize.
            if contour.success and contour.mask is not None:
                geom_shapely = polygonize_contour(
                    contour.mask,
                    (det.x1, det.y1),
                    transform,
                    crs,
                    simplify_tolerance=cfg.simplify_tolerance,
                )
            else:
                geom_shapely = None

            if geom_shapely is not None and not geom_shapely.is_empty:
                geometry_source = GeometrySource.derived_contour
                geometry_quality = GeometryQuality.approximate
                geojson_geom = geom_shapely.__geo_interface__
            else:
                # Fallback to bbox polygon.
                geometry_source = GeometrySource.bbox_fallback
                geometry_quality = GeometryQuality.fallback
                if "contour_extraction_failed" not in flags:
                    flags.append("contour_extraction_failed")
                geojson_geom = bbox_to_polygon(det.x1, det.y1, det.x2, det.y2, transform)
                # Create a shapely geometry from the GeoJSON for metrics.
                from shapely.geometry import shape as shapely_shape

                geom_shapely = shapely_shape(geojson_geom)

            # Geometry metrics.
            metrics = compute_geometry_metrics(geom_shapely, crs)

            # Envelope area (the full YOLO bbox).
            bbox_geojson = bbox_to_polygon(det.x1, det.y1, det.x2, det.y2, transform)
            from shapely.geometry import shape as shapely_shape2

            bbox_shapely = shapely_shape2(bbox_geojson)
            from oilspill.pipeline.vectorize import polygon_area_km2

            envelope_area = polygon_area_km2(bbox_shapely, crs)

            # Derived contour area.
            if geometry_source == GeometrySource.derived_contour:
                contour_area = metrics["area_km2"]
            else:
                contour_area = None

            # Land overlap.
            land_frac = 0.0
            if land_geometries:
                land_frac = screen_land_overlap(geom_shapely, land_geometries)
                if land_frac > 0.5:
                    flags.append("land_overlap_high")
                elif land_frac > 0.1:
                    flags.append("land_overlap_partial")

            # Environmental flags.
            ctx = env_context or {}
            if ctx.get("wind_speed_ms") is None:
                flags.append("wind_data_unavailable")
            elif ctx["wind_speed_ms"] < cfg.low_wind_threshold:
                flags.append("possible_low_wind_lookalike")

            if ctx.get("optical_corroboration") is None:
                flags.append("optical_confirmation_unavailable")

            # Investigation confidence.
            inv_conf, breakdown = compute_investigation_confidence(
                det.confidence,
                flags,
                land_overlap_frac=land_frac,
                contour_success=contour.success,
                env_context=env_context,
                low_wind_threshold=cfg.low_wind_threshold,
            )

            candidates.append(
                CandidateResult(
                    detector_type=DetectorType.yolo_mvp,
                    model_id=model_id,
                    model_confidence=det.confidence,
                    investigation_confidence=inv_conf,
                    investigation_confidence_breakdown=breakdown,
                    geometry_source=geometry_source,
                    geometry_quality=geometry_quality,
                    quality_flags=flags,
                    geometry=geojson_geom,
                    candidate_envelope_area_km2=envelope_area,
                    derived_contour_area_km2=contour_area,
                    centroid=(metrics["centroid_lon"], metrics["centroid_lat"]),
                    perimeter_km=metrics["perimeter_km"],
                    major_axis_km=metrics["major_axis_km"],
                    minor_axis_km=metrics["minor_axis_km"],
                    elongation=metrics["elongation"],
                    orientation_degrees=metrics["orientation_degrees"],
                    tile_provenance=[det.tile_index],
                )
            )

        # Scene bbox.
        h, w = sar_db.shape[:2]
        corners = [transform * (0, 0), transform * (w, 0), transform * (w, h), transform * (0, h)]
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        scene_bbox = [min(lons), min(lats), max(lons), max(lats)]

        return DetectionOutput(
            detector_type=DetectorType.yolo_mvp,
            model_id=model_id,
            scene_id=scene_id,
            crs=str(crs),
            bbox=scene_bbox,
            candidates=candidates,
            scene_metadata={
                "tile_size": cfg.tile_size,
                "tile_overlap": cfg.tile_overlap,
                "conf_threshold": cfg.conf_threshold,
                "iou_threshold": cfg.iou_threshold,
                "db_window": [cfg.db_min, cfg.db_max],
                "num_raw_detections_before_nms": len(detections),
            },
        )


__all__ = [
    "ContourResult",
    "RawDetection",
    "TileInfo",
    "YoloDetector",
    "YoloDetectorConfig",
    "bbox_to_polygon",
    "compute_geometry_metrics",
    "compute_investigation_confidence",
    "extract_contour",
    "generate_tiles",
    "global_nms",
    "polygonize_contour",
    "run_tiled_yolo",
    "sar_to_yolo_image",
    "screen_land_overlap",
]
