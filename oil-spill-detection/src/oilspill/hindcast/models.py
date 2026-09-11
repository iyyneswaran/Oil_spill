"""Validated, dependency-light input contracts for backward drift simulation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class CurrentOilSlick(BaseModel):
    """Centroid and detection metadata of the slick being backtracked."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    detection_time: datetime
    slick_area_km2: float = Field(gt=0)

    @field_validator("detection_time")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("detection_time must include a timezone offset")
        return value


class HistoricalVector(BaseModel):
    """A regional historical east/north vector observation in metres per second.

    The observations must be for the ocean region being hindcast.  A vector is
    linearly interpolated in time between adjacent samples.  This small contract
    deliberately does not pretend to retrieve or infer oceanographic data.
    """

    timestamp: datetime
    eastward_ms: float
    northward_ms: float

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value


class HindcastConfig(BaseModel):
    """Numerical controls for the ensemble Lagrangian backtrack."""

    duration_hours: float = Field(default=24.0, gt=0, le=24 * 30)
    step_minutes: float = Field(default=30.0, gt=0, le=120)
    particle_count: int = Field(default=250, ge=10, le=10_000)
    windage: float = Field(default=0.03, ge=0, le=0.2)
    horizontal_diffusivity_m2_s: float = Field(default=5.0, ge=0, le=10_000)
    uncertainty_confidence: float = Field(default=0.9, gt=0, lt=1)
    random_seed: int | None = 0


class HindcastInput(BaseModel):
    """One detected slick and historical forcing used to rewind its transport."""

    current_oil_slick: CurrentOilSlick
    ocean_currents: list[HistoricalVector] = Field(min_length=2)
    winds: list[HistoricalVector] = Field(min_length=2)
    config: HindcastConfig = Field(default_factory=HindcastConfig)

    @field_validator("ocean_currents", "winds")
    @classmethod
    def _sort_and_require_distinct_timestamps(
        cls, values: list[HistoricalVector]
    ) -> list[HistoricalVector]:
        ordered = sorted(values, key=lambda value: value.timestamp)
        timestamps = [value.timestamp for value in ordered]
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("historical vector timestamps must be unique")
        return ordered

    @model_validator(mode="after")
    def _require_forcing_over_full_hindcast_window(self) -> HindcastInput:
        start = (
            self.current_oil_slick.detection_time.timestamp() - self.config.duration_hours * 3600
        )
        end = self.current_oil_slick.detection_time.timestamp()
        for label, vectors in (("ocean_currents", self.ocean_currents), ("winds", self.winds)):
            first = vectors[0].timestamp.timestamp()
            last = vectors[-1].timestamp.timestamp()
            if first > start or last < end:
                raise ValueError(
                    f"{label} must cover the full hindcast window "
                    "from detection_time - duration_hours through detection_time"
                )
        return self
