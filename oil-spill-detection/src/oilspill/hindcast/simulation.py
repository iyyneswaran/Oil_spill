"""Multi-particle Lagrangian backward transport and portable output serializers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import pi, sqrt
from typing import Any

import numpy as np

from oilspill.hindcast.models import CurrentOilSlick, HindcastInput, HistoricalVector

_METRES_PER_DEGREE_LAT = 111_195.0


def _isoformat(value: datetime) -> str:
    """Return an unambiguous RFC 3339 timestamp, normalised to UTC."""
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone offset")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _interpolate(vectors: list[HistoricalVector], at: datetime) -> tuple[float, float]:
    """Linearly interpolate an east/north historical vector at ``at``."""
    target = at.timestamp()
    samples = np.asarray([item.timestamp.timestamp() for item in vectors], dtype=float)
    east = np.asarray([item.eastward_ms for item in vectors], dtype=float)
    north = np.asarray([item.northward_ms for item in vectors], dtype=float)
    return float(np.interp(target, samples, east)), float(np.interp(target, samples, north))


def _move(
    latitudes: np.ndarray, longitudes: np.ndarray, east_m: np.ndarray, north_m: np.ndarray
) -> None:
    """Apply small east/north displacements in-place on WGS84 coordinates."""
    latitudes += north_m / _METRES_PER_DEGREE_LAT
    # At the poles longitude is undefined; the simulation cannot usefully be
    # interpreted there, but this guard keeps the numerical transform finite.
    metres_per_degree_lon = _METRES_PER_DEGREE_LAT * np.maximum(
        np.abs(np.cos(np.deg2rad(latitudes))), 1e-8
    )
    longitudes += east_m / metres_per_degree_lon
    longitudes[:] = ((longitudes + 180.0) % 360.0) - 180.0
    np.clip(latitudes, -89.999999, 89.999999, out=latitudes)


def _initial_particles(
    slick: CurrentOilSlick, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Seed particles across an equal-area disk with the detected slick's area."""
    radius_m = sqrt(slick.slick_area_km2 * 1_000_000 / pi)
    radial = radius_m * np.sqrt(rng.random(count))
    angle = rng.uniform(0.0, 2.0 * pi, count)
    latitudes = np.full(count, slick.latitude, dtype=float)
    longitudes = np.full(count, slick.longitude, dtype=float)
    _move(latitudes, longitudes, radial * np.cos(angle), radial * np.sin(angle))
    return latitudes, longitudes


@dataclass(frozen=True)
class HindcastResult:
    """The six requested output values and their map-ready representation."""

    current_oil_slick_location: dict[str, float]
    detection_time: str
    backward_trajectories: list[dict[str, Any]]
    probable_spill_origin: dict[str, float]
    estimated_spill_time: str
    origin_probability: float

    def to_dict(self) -> dict[str, Any]:
        """Return exactly the machine-readable hindcast output contract."""
        return {
            "current_oil_slick_location": self.current_oil_slick_location,
            "detection_time": self.detection_time,
            "backward_trajectories": self.backward_trajectories,
            "probable_spill_origin": self.probable_spill_origin,
            "estimated_spill_time": self.estimated_spill_time,
            "origin_probability": self.origin_probability,
        }

    def to_geojson(self) -> dict[str, Any]:
        """Return a GeoJSON FeatureCollection for a map, with no AIS or forecast data."""
        features: list[dict[str, Any]] = [
            {
                "type": "Feature",
                "properties": {
                    "result": "current_oil_slick",
                    "detection_time": self.detection_time,
                    "slick_area_km2": self.current_oil_slick_location["slick_area_km2"],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        self.current_oil_slick_location["longitude"],
                        self.current_oil_slick_location["latitude"],
                    ],
                },
            }
        ]
        for trajectory in self.backward_trajectories:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "result": "backward_trajectory",
                        "particle_id": trajectory["particle_id"],
                    },
                    "geometry": {"type": "LineString", "coordinates": trajectory["coordinates"]},
                }
            )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "result": "probable_spill_origin",
                    "estimated_spill_time": self.estimated_spill_time,
                    "origin_probability": self.origin_probability,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        self.probable_spill_origin["longitude"],
                        self.probable_spill_origin["latitude"],
                    ],
                },
            }
        )
        return {"type": "FeatureCollection", "features": features}


def run_hindcast(request: HindcastInput) -> HindcastResult:
    """Rewind a slick through historical regional currents and winds.

    Each particle is seeded over the detected slick area.  At every backward
    time step it moves against ``current + windage * wind`` and receives an
    isotropic random-walk displacement with variance ``2*K*dt``.  The terminal
    ensemble forms an origin probability zone rather than claiming an exact
    discharge point.
    """
    config = request.config
    slick = request.current_oil_slick
    step_seconds = config.step_minutes * 60.0
    requested_seconds = config.duration_hours * 3600.0
    full_steps, final_seconds = divmod(requested_seconds, step_seconds)
    step_sizes = [step_seconds] * int(full_steps)
    if final_seconds > 1e-9:
        step_sizes.append(final_seconds)

    rng = np.random.default_rng(config.random_seed)
    latitudes, longitudes = _initial_particles(slick, config.particle_count, rng)
    trajectories = [
        {"particle_id": index, "coordinates": [[float(lon), float(lat)]]}
        for index, (lat, lon) in enumerate(zip(latitudes, longitudes, strict=True))
    ]
    now = slick.detection_time
    for dt in step_sizes:
        midpoint = now - timedelta(seconds=dt / 2.0)
        current_east, current_north = _interpolate(request.ocean_currents, midpoint)
        wind_east, wind_north = _interpolate(request.winds, midpoint)
        # Reverse the forward transport to obtain the immediately preceding
        # position.  Diffusion is symmetric in time, so a new random-walk term
        # represents unresolved turbulent transport while backtracking.
        eastward_ms = current_east + config.windage * wind_east
        northward_ms = current_north + config.windage * wind_north
        sigma_m = sqrt(2.0 * config.horizontal_diffusivity_m2_s * dt)
        _move(
            latitudes,
            longitudes,
            -eastward_ms * dt + rng.normal(0.0, sigma_m, config.particle_count),
            -northward_ms * dt + rng.normal(0.0, sigma_m, config.particle_count),
        )
        now -= timedelta(seconds=dt)
        for trajectory, lat, lon in zip(trajectories, latitudes, longitudes, strict=True):
            trajectory["coordinates"].append([float(lon), float(lat)])

    # Longitude is circular, so compute the ensemble centre with circular means.
    origin_lat = float(np.mean(latitudes))
    origin_lon = float(
        np.degrees(
            np.arctan2(
                np.mean(np.sin(np.deg2rad(longitudes))), np.mean(np.cos(np.deg2rad(longitudes)))
            )
        )
    )
    estimated_spill_time = slick.detection_time - timedelta(seconds=requested_seconds)
    probability = round(config.uncertainty_confidence, 6)

    return HindcastResult(
        current_oil_slick_location={
            "latitude": slick.latitude,
            "longitude": slick.longitude,
            "slick_area_km2": slick.slick_area_km2,
        },
        detection_time=_isoformat(slick.detection_time),
        backward_trajectories=trajectories,
        # The origin is calculated from every particle's terminal, backtracked
        # position.  Longitude uses a circular mean so a distribution spanning
        # the antimeridian remains centred correctly.
        probable_spill_origin={"latitude": origin_lat, "longitude": origin_lon},
        estimated_spill_time=_isoformat(estimated_spill_time),
        origin_probability=probability,
    )


__all__ = ["HindcastResult", "run_hindcast"]
