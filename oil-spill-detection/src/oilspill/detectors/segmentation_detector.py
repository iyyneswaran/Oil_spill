"""Thin wrapper that adapts the existing segmentation pipeline to the shared contract.

This module does NOT replace or duplicate the segmentation pipeline.  It simply
wraps the output of :func:`oilspill.pipeline.detect.run_detection` into the
:class:`~oilspill.detectors.contracts.DetectionOutput` contract so both detector
modes can be consumed uniformly by the API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oilspill.detectors.contracts import (
    CandidateResult,
    ConfidenceAdjustment,
    DetectionOutput,
    DetectorType,
    GeometryQuality,
    GeometrySource,
)

if TYPE_CHECKING:
    import geopandas as gpd


def segmentation_result_to_contract(
    gdf: gpd.GeoDataFrame,
    *,
    model_id: str,
    scene_id: str,
    crs: str,
    bbox: list[float],
) -> DetectionOutput:
    """Wrap a segmentation GeoDataFrame into the shared detection contract.

    Parameters
    ----------
    gdf:
        Oil-spill polygons from :func:`oilspill.pipeline.vectorize.vectorize_oil`,
        with ``area_km2`` and optional ``mean_confidence``/``max_confidence``
        columns.
    model_id:
        Identifier for the segmentation model.
    scene_id:
        Scene identifier.
    crs:
        CRS string (e.g. ``"EPSG:4326"``).
    bbox:
        Scene bounding box ``[min_lon, min_lat, max_lon, max_lat]``.

    Returns
    -------
    DetectionOutput
    """
    import math

    from pyproj import CRS as PyprojCRS
    from pyproj import Geod, Transformer
    from shapely.geometry.base import BaseGeometry

    candidates: list[CandidateResult] = []
    geod = Geod(ellps="WGS84")
    pyproj_crs = PyprojCRS.from_user_input(crs)

    for idx in range(len(gdf)):
        row = gdf.iloc[idx]
        geom: BaseGeometry = row.geometry  # type: ignore[assignment]
        area_km2: float = row.get("area_km2", 0.0)  # type: ignore[assignment]
        mean_conf: float = row.get("mean_confidence", 1.0)  # type: ignore[assignment]

        # Centroid.
        centroid = geom.centroid
        if not pyproj_crs.is_geographic:
            transformer = Transformer.from_crs(
                pyproj_crs, PyprojCRS.from_epsg(4326), always_xy=True
            )
            clon, clat = transformer.transform(centroid.x, centroid.y)
        else:
            clon, clat = centroid.x, centroid.y

        # Perimeter.
        if pyproj_crs.is_geographic:
            _, perimeter_m = geod.geometry_area_perimeter(geom)
            perimeter_km = abs(perimeter_m) / 1e3
        else:
            perimeter_km = float(geom.length) / 1e3

        # MRR metrics.
        mrr = geom.minimum_rotated_rectangle
        major_km = minor_km = 0.0
        elongation = 1.0
        orientation = 0.0
        if mrr is not None and not mrr.is_empty:
            coords = list(mrr.exterior.coords)
            if len(coords) >= 4:
                edges = []
                for i in range(len(coords) - 1):
                    dx = coords[i + 1][0] - coords[i][0]
                    dy = coords[i + 1][1] - coords[i][1]
                    length = math.sqrt(dx**2 + dy**2)
                    edges.append((length, dx, dy))
                edges.sort(key=lambda e: e[0], reverse=True)
                if pyproj_crs.is_geographic and len(edges) >= 2:
                    _, _, d1 = geod.inv(
                        coords[0][0], coords[0][1], coords[1][0], coords[1][1]
                    )
                    _, _, d2 = geod.inv(
                        coords[1][0], coords[1][1], coords[2][0], coords[2][1]
                    )
                    major_km = max(abs(d1), abs(d2)) / 1e3
                    minor_km = min(abs(d1), abs(d2)) / 1e3
                elif len(edges) >= 2:
                    major_km = edges[0][0] / 1e3
                    minor_km = edges[1][0] / 1e3
                elongation = major_km / minor_km if minor_km > 0 else 1.0
                if edges:
                    _, dx, dy = edges[0]
                    orientation = math.degrees(math.atan2(dx, dy)) % 180

        candidates.append(
            CandidateResult(
                detector_type=DetectorType.segmentation,
                model_id=model_id,
                model_confidence=float(mean_conf),
                investigation_confidence=float(mean_conf),
                investigation_confidence_breakdown=[
                    ConfidenceAdjustment(
                        reason="Segmentation model oil probability (no heuristic applied)",
                        delta=0.0,
                    )
                ],
                geometry_source=GeometrySource.segmentation_mask,
                geometry_quality=GeometryQuality.high,
                quality_flags=[],
                geometry=geom.__geo_interface__,
                candidate_envelope_area_km2=area_km2,
                derived_contour_area_km2=None,
                centroid=(clon, clat),
                perimeter_km=perimeter_km,
                major_axis_km=major_km,
                minor_axis_km=minor_km,
                elongation=elongation,
                orientation_degrees=orientation,
            )
        )

    return DetectionOutput(
        detector_type=DetectorType.segmentation,
        model_id=model_id,
        scene_id=scene_id,
        crs=crs,
        bbox=bbox,
        candidates=candidates,
    )


__all__ = ["segmentation_result_to_contract"]
