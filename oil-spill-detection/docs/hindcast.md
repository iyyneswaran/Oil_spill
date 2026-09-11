# Hindcast output

This component accepts one detected slick and caller-supplied historical, regional
current and wind vectors. It does not fetch data, invoke the detector, perform
AIS/vessel analysis, produce a forward forecast, or add a dashboard/API endpoint.

Run it with:

```powershell
uv run python scripts/hindcast.py --input hindcast-input.json --output-dir artifacts/hindcast
```

`hindcast-input.json` has this shape (timestamps require timezone offsets):

```json
{
  "current_oil_slick": {
    "latitude": -20.44,
    "longitude": 57.72,
    "detection_time": "2020-07-25T04:35:00Z",
    "slick_area_km2": 3.1
  },
  "ocean_currents": [
    {"timestamp": "2020-07-24T04:35:00Z", "eastward_ms": 0.12, "northward_ms": -0.04},
    {"timestamp": "2020-07-25T04:35:00Z", "eastward_ms": 0.09, "northward_ms": -0.02}
  ],
  "winds": [
    {"timestamp": "2020-07-24T04:35:00Z", "eastward_ms": 3.1, "northward_ms": 1.4},
    {"timestamp": "2020-07-25T04:35:00Z", "eastward_ms": 2.8, "northward_ms": 1.1}
  ],
  "config": {"duration_hours": 24, "particle_count": 250}
}
```

The input vectors must cover the full backtrack window. They are interpolated in
time. The simulation initializes particles over the slick area, rewinds each
against `current + windage * wind`, and applies diffusion. `hindcast.json`
contains only the six requested output values. `hindcast.geojson` contains a
current-slick point, particle lines (current to past), and the calculated origin
point. The reported `origin_probability` is an ensemble configuration value, not
a calibrated probability that a particular vessel discharged oil.
