"""Probability-based backward drift output for a detected oil slick.

This package is intentionally independent from the detection pipeline.  Callers
provide one detected slick plus historical wind and current observations; the
package returns only the hindcast result and its map-ready GeoJSON view.
"""

from oilspill.hindcast.models import (
    CurrentOilSlick,
    HindcastConfig,
    HindcastInput,
    HistoricalVector,
)
from oilspill.hindcast.simulation import HindcastResult, run_hindcast

__all__ = [
    "CurrentOilSlick",
    "HindcastConfig",
    "HindcastInput",
    "HindcastResult",
    "HistoricalVector",
    "run_hindcast",
]
