"""Pydantic request/response schemas for the inference API.

These mirror the JSON contract the frontend is built against exactly. Keeping the
schemas in one place lets FastAPI generate accurate OpenAPI docs at ``/docs`` and
gives the response handlers a single, typed source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --- /healthz ----------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: Literal["ok"] = "ok"


# --- /models -----------------------------------------------------------------


class PerClassMetrics(BaseModel):
    """Per-class evaluation metrics (any field may be ``null`` if unavailable)."""

    iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class ModelInfo(BaseModel):
    """One model's identity, headline metrics and availability."""

    id: str
    name: str
    oil_iou: float | None = None
    oil_recall: float | None = None
    mean_iou: float | None = None
    macro_f1: float | None = None
    pixel_accuracy: float | None = None
    per_class: dict[str, PerClassMetrics] = Field(default_factory=dict)
    available: bool = False


class ModelsResponse(BaseModel):
    """List of all known models."""

    models: list[ModelInfo]


# --- /samples ----------------------------------------------------------------


class SampleInfo(BaseModel):
    """A preloaded sample image and the URL to fetch its bytes."""

    id: str
    url: str


class SamplesResponse(BaseModel):
    """List of preloaded sample images."""

    samples: list[SampleInfo]


# --- /predict ----------------------------------------------------------------


class PredictResponse(BaseModel):
    """Segmentation result for a single uploaded image."""

    model: str
    width: int
    height: int
    class_percentages: dict[str, float]
    legend: dict[str, list[int]]
    mask_png: str
    overlay_png: str


# --- /jobs -------------------------------------------------------------------


class SceneJobRequest(BaseModel):
    """Request body for a full-scene detection job over an AOI/date range."""

    aoi: dict[str, Any]
    start: str
    end: str
    model: str | None = None
    detector: Literal["segmentation", "yolo_mvp"] = "segmentation"
    environmental_context: EnvironmentalContext | None = None


class EnvironmentalContext(BaseModel):
    """Optional environmental metadata supplied by the caller (not fetched)."""

    wind_speed_ms: float | None = None
    optical_corroboration: bool | None = None


class SceneJobResponse(BaseModel):
    """Acknowledgement returned when a scene job is accepted."""

    job_id: str
    status: Literal["queued"] = "queued"


class JobResult(BaseModel):
    """Summary statistics + vectorised polygons of a finished scene job."""

    num_oil_polygons: int
    total_oil_area_km2: float
    geojson: dict[str, Any]


class JobStatusResponse(BaseModel):
    """Current state of a scene job (polled by the frontend)."""

    job_id: str
    status: Literal["queued", "running", "done", "error"]
    detail: str | None = None
    result: JobResult | None = None


# --- YOLO MVP ----------------------------------------------------------------


class ConfidenceAdjustmentResponse(BaseModel):
    """One entry in the heuristic confidence breakdown."""

    reason: str
    delta: float


class YoloCandidateResponse(BaseModel):
    """A single YOLO MVP oil-spill candidate.

    ``model_confidence`` is the raw YOLO detector confidence.
    ``investigation_confidence`` is a **heuristic ranking score** (NOT a
    calibrated probability) — see ``investigation_confidence_breakdown`` for
    every adjustment applied.
    """

    detector_type: str
    model_id: str
    model_confidence: float
    investigation_confidence: float
    investigation_confidence_breakdown: list[ConfidenceAdjustmentResponse]
    geometry_source: str
    geometry_quality: str
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


class YoloDetectionResponse(BaseModel):
    """Scene-level YOLO MVP detection result."""

    detector_type: str
    model_id: str
    scene_id: str
    crs: str
    bbox: list[float]
    candidates: list[YoloCandidateResponse]
    scene_metadata: dict[str, Any] = Field(default_factory=dict)


class YoloStatusResponse(BaseModel):
    """YOLO detector availability status."""

    available: bool
    model_id: str | None = None
    detail: str


__all__ = [
    "ConfidenceAdjustmentResponse",
    "EnvironmentalContext",
    "HealthResponse",
    "JobResult",
    "JobStatusResponse",
    "ModelInfo",
    "ModelsResponse",
    "PerClassMetrics",
    "PredictResponse",
    "SampleInfo",
    "SamplesResponse",
    "SceneJobRequest",
    "SceneJobResponse",
    "YoloCandidateResponse",
    "YoloDetectionResponse",
    "YoloStatusResponse",
]
