"""Detector abstraction layer for oil-spill detection.

Provides a unified output contract and two concrete detector implementations:

* **Segmentation** — wraps the existing ONNX semantic-segmentation pipeline.
* **YOLO MVP** — a separate bounding-box detector with image-derived contour
  extraction, transparent confidence metadata, and explicit quality flags.

Both detectors produce :class:`~oilspill.detectors.contracts.DetectionOutput`
so the API and frontend can consume either uniformly.
"""

from __future__ import annotations

from oilspill.detectors.contracts import (
    CandidateResult,
    ConfidenceAdjustment,
    DetectionOutput,
    DetectorType,
    GeometryQuality,
    GeometrySource,
)

__all__ = [
    "CandidateResult",
    "ConfidenceAdjustment",
    "DetectionOutput",
    "DetectorType",
    "GeometryQuality",
    "GeometrySource",
]
