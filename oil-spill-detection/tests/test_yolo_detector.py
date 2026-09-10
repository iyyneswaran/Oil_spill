"""Tests for the YOLO MVP detector abstraction.

These tests run without the actual YOLO weights or an active network connection,
using a mocked model and synthetic arrays to verify tiling, NMS, contour extraction,
and confidence heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from oilspill.api.app import create_app
from oilspill.api.settings import Settings
from oilspill.detectors.yolo_detector import (
    RawDetection,
    YoloDetector,
    YoloDetectorConfig,
    compute_geometry_metrics,
    compute_investigation_confidence,
    extract_contour,
    generate_tiles,
    global_nms,
    polygonize_contour,
    sar_to_yolo_image,
    screen_land_overlap,
)
from rasterio.crs import CRS
from affine import Affine

# --- 1. Tile coordinate mapping ----------------------------------------------


def test_generate_tiles():
    # 2000x2000 image, 1024 tile size, 128 overlap. Stride = 896.
    # x/y stops: 0, 896, 1792 (clamped to 2000-1024 = 976)
    tiles = generate_tiles(2000, 2000, tile_size=1024, overlap=128)
    
    # 3x3 grid = 9 tiles total.
    assert len(tiles) == 9
    
    # First tile is at 0,0 and full size.
    assert tiles[0].y == 0
    assert tiles[0].x == 0
    assert tiles[0].h == 1024
    assert tiles[0].w == 1024
    
    # Check bottom-right tile is clamped correctly.
    bottom_right = tiles[-1]
    assert bottom_right.y + bottom_right.h == 2000
    assert bottom_right.x + bottom_right.w == 2000


# --- 2. Cross-tile NMS -------------------------------------------------------


def test_global_nms():
    # Two identical boxes, different confidences
    dets = [
        RawDetection(0, 0, 100, 100, 0.9, 0, 0),
        RawDetection(0, 0, 100, 100, 0.8, 0, 1),
    ]
    kept = global_nms(dets, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.9

    # Two disjoint boxes
    dets2 = [
        RawDetection(0, 0, 100, 100, 0.9, 0, 0),
        RawDetection(200, 200, 300, 300, 0.8, 0, 1),
    ]
    kept2 = global_nms(dets2, iou_threshold=0.5)
    assert len(kept2) == 2


# --- 3. Land-overlap rejection -----------------------------------------------


def test_screen_land_overlap():
    candidate = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    land = Polygon([(5, 0), (5, 10), (15, 10), (15, 0), (5, 0)])
    
    # Exactly half the candidate overlaps land.
    frac = screen_land_overlap(candidate, [land])
    assert frac == 0.5


# --- 4. SAR-to-YOLO renderer -------------------------------------------------


def test_sar_to_yolo_image():
    # dB window is -25 to 0.
    # -25 should map to 0, 0 should map to 255.
    db_img = np.array([[-30.0, -25.0], [-12.5, 0.0], [5.0, 10.0]], dtype=np.float32)
    rgb = sar_to_yolo_image(db_img, db_min=-25.0, db_max=0.0)
    
    assert rgb.shape == (3, 2, 3)
    assert rgb.dtype == np.uint8
    
    # Check scaling.
    assert rgb[0, 0, 0] == 0    # Clipped low
    assert rgb[0, 1, 0] == 0    # Exact min
    assert rgb[1, 0, 0] == 127  # Midpoint
    assert rgb[1, 1, 0] == 255  # Exact max
    assert rgb[2, 0, 0] == 255  # Clipped high


# --- 5. Contour extraction on synthetic array --------------------------------


def test_extract_contour_success():
    # Create a 100x100 synthetic dB array. Background ~0 dB (sea), blob ~-20 dB (oil).
    sar_db = np.zeros((100, 100), dtype=np.float32)
    sar_db[40:60, 40:60] = -20.0
    
    bbox = (20, 20, 80, 80)
    result = extract_contour(sar_db, bbox, min_pixels=10)
    
    assert result.success is True
    assert result.mask is not None
    assert result.component_darkness < -10.0
    assert result.component_area_px == 400


def test_extract_contour_fallback():
    # Uniform background -> Otsu fails to find a distinct dark region.
    sar_db = np.zeros((100, 100), dtype=np.float32)
    bbox = (20, 20, 80, 80)
    
    result = extract_contour(sar_db, bbox, min_pixels=10)
    
    assert result.success is False
    assert result.mask is None
    assert "uniform_sar_crop" in result.quality_flags


# --- 6. Quality flags and heuristic confidence -------------------------------


def test_compute_investigation_confidence():
    raw_conf = 0.8
    
    # 1. Base case
    score, breakdown = compute_investigation_confidence(raw_conf, [], 0.0, True, {})
    assert score == 0.8
    assert "Raw model confidence" in breakdown[0].reason
    
    # 2. Land overlap penalty (>50%)
    score2, _ = compute_investigation_confidence(raw_conf, [], 0.6, True, {})
    assert score2 == 0.4  # 0.8 - 0.4
    
    # 3. Low wind penalty
    score3, _ = compute_investigation_confidence(raw_conf, [], 0.0, True, {"wind_speed_ms": 1.0})
    assert score3 == 0.7  # 0.8 - 0.1
    
    # 4. Optical corroboration bonus
    score4, _ = compute_investigation_confidence(raw_conf, [], 0.0, True, {"optical_corroboration": True})
    assert score4 == 0.9  # 0.8 + 0.1


# --- 7. Geometry metrics -----------------------------------------------------


def test_compute_geometry_metrics():
    # 1000m x 2000m rectangle in a projected metric CRS.
    poly = Polygon([(0, 0), (0, 1000), (2000, 1000), (2000, 0), (0, 0)])
    crs = CRS.from_epsg(32631)  # UTM zone 31N
    
    metrics = compute_geometry_metrics(poly, crs)
    
    # Area = 2 km^2
    assert metrics["area_km2"] == pytest.approx(2.0, abs=1e-4)
    # Perimeter = 6 km
    assert metrics["perimeter_km"] == pytest.approx(6.0, abs=1e-4)
    # Major = 2 km, Minor = 1 km, Elongation = 2.0
    assert metrics["major_axis_km"] == pytest.approx(2.0, abs=1e-4)
    assert metrics["minor_axis_km"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["elongation"] == pytest.approx(2.0, abs=1e-4)


# --- 8. API endpoints and missing weights ------------------------------------


def test_api_yolo_status_missing(tmp_path: Path):
    cfg = Settings(
        yolo_weights=tmp_path / "does_not_exist.pt",
        results_dir=tmp_path / "docs/results",
        onnx_dir=tmp_path / "artifacts/exports",
        default_onnx=tmp_path / "artifacts/exports/model.onnx",
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        # /healthz still 200
        assert client.get("/healthz").status_code == 200
        
        # /yolo/status correctly reflects missing weights
        resp = client.get("/yolo/status")
        assert resp.status_code == 200
        assert resp.json()["available"] is False
        
        # YOLO job submission 503s gracefully
        body = {
            "aoi": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]},
            "start": "2024-01-01",
            "end": "2024-01-31",
            "detector": "yolo_mvp",
        }
        job_resp = client.post("/jobs/scene", json=body)
        assert job_resp.status_code == 503
        assert "YOLO detector unavailable" in job_resp.text
