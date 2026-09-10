"""Shared output contract for all oil-spill detectors.

Every detector — whether the YOLO MVP bounding-box detector or the ONNX semantic
segmentation model — emits :class:`DetectionOutput` containing a list of
:class:`CandidateResult` objects.  The contract enforces transparent metadata about
the detection source, geometry derivation method, quality flags, and a documented
heuristic confidence breakdown so downstream consumers (the API, the frontend, and
any future analysis) never confuse raw model output with calibrated probabilities
or approximate image-derived contours with production-grade segmentation masks.

This file is deliberately dependency-light (only stdlib + typing) so it can be
imported anywhere without pulling in heavy geospatial or ML packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DetectorType(str, Enum):
    """Identifies which detector produced a result."""

    yolo_mvp = "yolo_mvp"
    segmentation = "segmentation"


class GeometrySource(str, Enum):
    """How the candidate polygon geometry was derived."""

    derived_contour = "derived_contour"
    """Dark-region contour extracted from the SAR image within a YOLO bbox."""

    bbox_fallback = "bbox_fallback"
    """Georeferenced YOLO bounding box used when contour extraction failed."""

    segmentation_mask = "segmentation_mask"
    """Polygon vectorised from a semantic-segmentation class mask."""


class GeometryQuality(str, Enum):
    """Qualitative reliability of the candidate geometry."""

    approximate = "approximate"
    """Image-derived contour: shape is plausible but not pixel-exact."""

    fallback = "fallback"
    """Bounding-box rectangle: only an envelope, not a shape estimate."""

    high = "high"
    """Segmentation mask: pixel-level class assignment."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceAdjustment:
    """One entry in the heuristic investigation-confidence breakdown.

    Every adjustment carries a human-readable *reason* and a signed *delta*
    (positive = bonus, negative = penalty).  The full list is returned to the
    caller so every score adjustment is transparent and auditable.
    """

    reason: str
    delta: float


@dataclass
class CandidateResult:
    """A single oil-spill candidate produced by any detector.

    Attributes
    ----------
    detector_type:
        Which detector produced this candidate.
    model_id:
        Identifier of the model (e.g. checkpoint filename or ONNX model name).
    model_confidence:
        Raw detector confidence.  For YOLO this is the objectness × class
        confidence; for segmentation it is the per-polygon mean oil probability.
    investigation_confidence:
        A *heuristic ranking score* (0–1) that applies visible penalties and
        bonuses to the raw confidence.  **This is NOT a calibrated probability.**
        It is explicitly labelled as a heuristic score and returned with a
        human-readable breakdown of every adjustment.
    investigation_confidence_breakdown:
        Ordered list of adjustments that produced ``investigation_confidence``.
    geometry_source:
        How the polygon geometry was obtained.
    geometry_quality:
        Qualitative reliability of the geometry.
    quality_flags:
        Transparent list of reasons, warnings, or informational tags about this
        candidate (e.g. ``"contour_extraction_failed"``, ``"land_overlap_high"``,
        ``"wind_data_unavailable"``).
    geometry:
        GeoJSON-compatible geometry dict (Polygon or MultiPolygon).
    candidate_envelope_area_km2:
        Area of the YOLO bounding-box envelope in km².  For segmentation
        candidates this equals the polygon area.
    derived_contour_area_km2:
        Area of the image-derived contour polygon in km², or ``None`` when the
        geometry is a bbox fallback or segmentation mask.
    centroid:
        ``(longitude, latitude)`` of the polygon centroid in EPSG:4326.
    perimeter_km:
        Geodesic perimeter of the polygon in km.
    major_axis_km:
        Length of the major axis of the minimum rotated rectangle, in km.
    minor_axis_km:
        Length of the minor axis of the minimum rotated rectangle, in km.
    elongation:
        Ratio ``major_axis / minor_axis`` (≥ 1).  ``1.0`` means circular.
    orientation_degrees:
        Orientation of the major axis in degrees from north (0–180).
    tile_provenance:
        Optional list of tile indices that contributed this detection (YOLO only).
    component_count:
        Number of viable dark connected components seen in the YOLO box, when
        contour extraction ran. It is an approximate fragmentation indicator.
    """

    detector_type: DetectorType
    model_id: str
    model_confidence: float
    investigation_confidence: float
    investigation_confidence_breakdown: list[ConfidenceAdjustment]
    geometry_source: GeometrySource
    geometry_quality: GeometryQuality
    quality_flags: list[str]
    geometry: dict[str, Any]
    candidate_envelope_area_km2: float
    derived_contour_area_km2: float | None
    centroid: tuple[float, float]
    perimeter_km: float
    major_axis_km: float
    minor_axis_km: float
    elongation: float
    orientation_degrees: float
    tile_provenance: list[int] = field(default_factory=list)
    component_count: int | None = None


@dataclass
class DetectionOutput:
    """Scene-level result produced by any detector.

    Wraps a list of :class:`CandidateResult` objects with scene-level metadata.
    """

    detector_type: DetectorType
    model_id: str
    scene_id: str
    crs: str
    bbox: list[float]
    candidates: list[CandidateResult]
    scene_metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CandidateResult",
    "ConfidenceAdjustment",
    "DetectionOutput",
    "DetectorType",
    "GeometryQuality",
    "GeometrySource",
]
