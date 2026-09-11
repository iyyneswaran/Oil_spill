"""Tests for the isolated probability-based hindcast output component."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from oilspill.hindcast import (
    CurrentOilSlick,
    HindcastConfig,
    HindcastInput,
    HistoricalVector,
    run_hindcast,
)


def _request() -> HindcastInput:
    return HindcastInput(
        current_oil_slick=CurrentOilSlick(
            latitude=10.0,
            longitude=70.0,
            detection_time=datetime.fromisoformat("2026-01-02T12:00:00+00:00"),
            slick_area_km2=1.0,
        ),
        ocean_currents=[
            HistoricalVector(
                timestamp=datetime.fromisoformat("2026-01-02T10:00:00+00:00"),
                eastward_ms=1.0,
                northward_ms=0.0,
            ),
            HistoricalVector(
                timestamp=datetime.fromisoformat("2026-01-02T12:00:00+00:00"),
                eastward_ms=1.0,
                northward_ms=0.0,
            ),
        ],
        winds=[
            HistoricalVector(
                timestamp=datetime.fromisoformat("2026-01-02T10:00:00+00:00"),
                eastward_ms=0.0,
                northward_ms=0.0,
            ),
            HistoricalVector(
                timestamp=datetime.fromisoformat("2026-01-02T12:00:00+00:00"),
                eastward_ms=0.0,
                northward_ms=0.0,
            ),
        ],
        config=HindcastConfig(
            duration_hours=2,
            step_minutes=60,
            particle_count=20,
            horizontal_diffusivity_m2_s=0,
            random_seed=7,
        ),
    )


def test_hindcast_rewinds_against_current_and_returns_only_requested_fields() -> None:
    result = run_hindcast(_request())

    assert list(result.to_dict()) == [
        "current_oil_slick_location",
        "detection_time",
        "backward_trajectories",
        "probable_spill_origin",
        "estimated_spill_time",
        "origin_probability",
    ]
    assert result.probable_spill_origin["longitude"] < 70.0
    assert result.probable_spill_origin["latitude"] != 10.0
    terminal_positions = np.asarray(
        [trajectory["coordinates"][-1] for trajectory in result.backward_trajectories]
    )
    assert result.probable_spill_origin["latitude"] == pytest.approx(
        terminal_positions[:, 1].mean()
    )
    assert result.probable_spill_origin["longitude"] == pytest.approx(
        terminal_positions[:, 0].mean()
    )
    assert result.estimated_spill_time == "2026-01-02T10:00:00Z"
    assert len(result.backward_trajectories) == 20
    assert all(len(item["coordinates"]) == 3 for item in result.backward_trajectories)
    assert result.origin_probability == 0.9


def test_geojson_has_slick_trajectories_and_calculated_origin() -> None:
    geojson = run_hindcast(_request()).to_geojson()

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 22
    assert geojson["features"][0]["properties"]["result"] == "current_oil_slick"
    assert geojson["features"][-1]["properties"]["result"] == "probable_spill_origin"
    assert geojson["features"][-1]["geometry"]["type"] == "Point"


def test_hindcast_requires_historical_data_for_entire_backtrack_window() -> None:
    with pytest.raises(ValidationError, match="must cover the full hindcast window"):
        HindcastInput(
            current_oil_slick=CurrentOilSlick(
                latitude=10.0,
                longitude=70.0,
                detection_time=datetime.fromisoformat("2026-01-02T12:00:00+00:00"),
                slick_area_km2=1.0,
            ),
            ocean_currents=[
                HistoricalVector(
                    timestamp=datetime.fromisoformat("2026-01-02T11:00:00+00:00"),
                    eastward_ms=0,
                    northward_ms=0,
                ),
                HistoricalVector(
                    timestamp=datetime.fromisoformat("2026-01-02T12:00:00+00:00"),
                    eastward_ms=0,
                    northward_ms=0,
                ),
            ],
            winds=[
                HistoricalVector(
                    timestamp=datetime.fromisoformat("2026-01-02T10:00:00+00:00"),
                    eastward_ms=0,
                    northward_ms=0,
                ),
                HistoricalVector(
                    timestamp=datetime.fromisoformat("2026-01-02T12:00:00+00:00"),
                    eastward_ms=0,
                    northward_ms=0,
                ),
            ],
            config=HindcastConfig(duration_hours=2),
        )


def test_hindcast_rejects_timestamp_without_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone offset"):
        CurrentOilSlick(
            latitude=10.0,
            longitude=70.0,
            detection_time=datetime(2026, 1, 2, 12),
            slick_area_km2=1.0,
        )
