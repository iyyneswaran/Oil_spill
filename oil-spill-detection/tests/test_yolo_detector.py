"""Tests for the YOLO MVP detector abstraction.

These tests run without the actual YOLO weights or an active network connection,
using a mocked model and synthetic arrays to verify tiling, NMS, contour extraction,
and confidence heuristics.
"""

from __future__ import annotations

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
    run_tiled_yolo,
    sar_to_yolo_image,
    screen_land_overlap,
)
from oilspill.pipeline.detect import detect_yolo_from_safe
from oilspill.pipeline.preprocess import CalibratedScene
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


class _FakeTensor:
    """Minimal torch-like value used by the no-Ultralytics inference mock."""

    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeBoxes:
    def __init__(self, xyxy: list[list[float]], confidence: list[float]) -> None:
        self.xyxy = [_FakeTensor(box) for box in xyxy]
        self.conf = [_FakeTensor(value) for value in confidence]
        self.cls = [_FakeTensor(0) for _ in xyxy]

    def __len__(self) -> int:
        return len(self.xyxy)


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _TiledMockYolo:
    """Returns the same box from two overlapping tiles and one clipped box."""

    def predict(self, crop: np.ndarray, **_kwargs: object) -> list[_FakeResult]:
        x_origin = int(crop[0, 0, 0])
        if x_origin == 0:
            return [_FakeResult(_FakeBoxes([[6, 2, 8, 6], [-2, 0, 2, 2]], [0.9, 0.7]))]
        return [_FakeResult(_FakeBoxes([[0, 2, 2, 6]], [0.8]))]


def test_tiled_yolo_remaps_clips_and_merges_provenance() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(12, dtype=np.uint8)[None, :]
    detections = run_tiled_yolo(
        _TiledMockYolo(), image, tile_size=8, tile_overlap=2, conf=0.1, iou=0.5
    )

    assert len(detections) == 2
    duplicate = next(det for det in detections if det.x1 == 6)
    assert (duplicate.x1, duplicate.y1, duplicate.x2, duplicate.y2) == (6, 2, 8, 6)
    assert duplicate.tile_indices == [0, 1]
    clipped = next(det for det in detections if det.x1 == 0)
    assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (0, 0, 2, 2)


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


def test_sar_to_yolo_image_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="db_min < db_max"):
        sar_to_yolo_image(np.zeros((2, 2), dtype=np.float32), db_min=0.0, db_max=0.0)


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
    # Morphological opening/closing may shave boundary pixels, so accept the
    # component within a small tolerance of the exact 20x20 = 400 px blob.
    assert result.component_area_px == pytest.approx(400, abs=10)


def test_extract_contour_fallback():
    # Uniform background -> Otsu fails to find a distinct dark region.
    sar_db = np.zeros((100, 100), dtype=np.float32)
    bbox = (20, 20, 80, 80)
    
    result = extract_contour(sar_db, bbox, min_pixels=10)
    
    assert result.success is False
    assert result.mask is None
    assert "uniform_sar_crop" in result.quality_flags


def test_extract_contour_handles_empty_and_invalid_crops() -> None:
    empty = extract_contour(np.zeros((5, 5), dtype=np.float32), (8, 8, 10, 10))
    assert empty.success is False
    assert "empty_bbox_crop" in empty.quality_flags
    invalid = extract_contour(np.full((5, 5), np.nan, dtype=np.float32), (0, 0, 5, 5))
    assert invalid.success is False
    assert "invalid_sar_crop" in invalid.quality_flags


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
    assert score3 == pytest.approx(0.7)  # 0.8 - 0.1
    
    # 4. Optical corroboration bonus
    score4, _ = compute_investigation_confidence(
        raw_conf, [], 0.0, True, {"optical_corroboration": True}
    )
    assert score4 == 0.9  # 0.8 + 0.1

    # 5. Explicit optical conflict is distinct from unavailable context.
    score5, breakdown5 = compute_investigation_confidence(
        raw_conf, [], 0.0, True, {"optical_conflict": True}
    )
    assert score5 == pytest.approx(0.7)
    assert any("conflicts" in item.reason for item in breakdown5)


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


def test_detector_emits_wgs84_contour_contract_and_land_flags() -> None:
    """The detector must expose portable GeoJSON, not UTM scene coordinates."""
    cfg = YoloDetectorConfig(tile_size=100, tile_overlap=10, contour_min_pixels=10)
    detector = YoloDetector(cfg)

    class _OneBoxModel:
        names = {0: "oil"}

        def predict(self, _crop: np.ndarray, **_kwargs: object) -> list[_FakeResult]:
            return [_FakeResult(_FakeBoxes([[20, 20, 80, 80]], [0.8]))]

    detector._model = _OneBoxModel()
    sar = np.zeros((100, 100), dtype=np.float32)
    sar[40:60, 40:60] = -20.0
    land = np.zeros((100, 100), dtype=bool)
    land[40:60, 40:60] = True
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    output = detector.detect(sar, transform, CRS.from_epsg(32631), land_mask=land)

    assert output.detector_type.value == "yolo_mvp"
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    assert candidate.geometry_source.value == "derived_contour"
    assert candidate.geometry_quality.value == "approximate"
    assert candidate.derived_contour_area_km2 is not None
    assert candidate.component_count == 1
    assert "land_overlap_high" in candidate.quality_flags
    # UTM eastings never appear in GeoJSON coordinates after the required reprojection.
    lon, lat = candidate.geometry["coordinates"][0][0]
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90
    assert candidate.centroid[0] == pytest.approx(3.0, abs=0.1)


def test_detector_falls_back_to_bbox_for_uniform_crop() -> None:
    cfg = YoloDetectorConfig(tile_size=100, tile_overlap=10, contour_min_pixels=10)
    detector = YoloDetector(cfg)

    class _OneBoxModel:
        names = {0: "oil"}

        def predict(self, _crop: np.ndarray, **_kwargs: object) -> list[_FakeResult]:
            return [_FakeResult(_FakeBoxes([[20, 20, 80, 80]], [0.8]))]

    detector._model = _OneBoxModel()
    output = detector.detect(
        np.zeros((100, 100), dtype=np.float32),
        Affine.translation(0.0, 1.0) * Affine.scale(0.01, -0.01),
        CRS.from_epsg(4326),
    )
    candidate = output.candidates[0]
    assert candidate.geometry_source.value == "bbox_fallback"
    assert candidate.derived_contour_area_km2 is None
    assert "contour_extraction_failed" in candidate.quality_flags
    assert "wind_data_unavailable" in candidate.quality_flags
    assert "optical_confirmation_unavailable" in candidate.quality_flags


def test_detect_yolo_from_safe_reuses_filtered_db_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The service-facing helper must pass filtered dB, transform, CRS and land mask."""
    transform = Affine.translation(10.0, 20.0) * Affine.scale(0.1, -0.1)
    scene = CalibratedScene(
        sigma0=np.ones((4, 5), dtype=np.float64), transform=transform, crs=CRS.from_epsg(4326)
    )
    monkeypatch.setattr("oilspill.pipeline.detect.calibrate_safe", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        "oilspill.pipeline.detect.lee_filter", lambda values, **_kwargs: values * 2.0
    )
    monkeypatch.setattr(
        "oilspill.pipeline.detect.to_db", lambda values: values - 10.0
    )
    monkeypatch.setattr(
        "oilspill.pipeline.detect.land_mask_from_coastlines",
        lambda shape, *_args: np.ones(shape, dtype=bool),
    )

    class _CaptureDetector:
        def detect(
            self, db: np.ndarray, got_transform: Affine, got_crs: CRS, **kwargs: object
        ) -> str:
            assert np.all(db == -8.0)
            assert got_transform == transform
            assert got_crs == scene.crs
            assert np.asarray(kwargs["land_mask"]).all()
            assert kwargs["scene_id"] == "scene"
            assert kwargs["env_context"] == {"wind_speed_ms": 2.0}
            return "ok"

    assert (
        detect_yolo_from_safe(
            tmp_path / "scene.SAFE",
            _CaptureDetector(),  # type: ignore[arg-type]
            coastlines_path=tmp_path / "land.geojson",
            env_context={"wind_speed_ms": 2.0},
        )
        == "ok"
    )


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


def test_annotate_yolo_image():
    from oilspill.api.service import annotate_yolo_image
    from oilspill.detectors.yolo_detector import RawDetection
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        RawDetection(x1=10, y1=10, x2=20, y2=20, confidence=0.9, class_id=0, tile_index=0)
    ]
    data_uri = annotate_yolo_image(img, detections)
    
    assert data_uri.startswith("data:image/png;base64,")
