# YOLO MVP Detector

The YOLO MVP detector is an alternative inference pathway that generates oil-candidate bounding boxes using an Ultralytics YOLO model. It is designed to run alongside the primary semantic-segmentation pipeline, sharing the same API and frontend without contaminating the segmentation results.

## Detector Boundary

The project has two independent detector paths. `segmentation` is the default
five-class ONNX semantic-segmentation pathway (Sea Surface, Oil Spill,
Look-alike, Ship, Land). `yolo_mvp` is a separate Ultralytics detector using an
external, one-class `oil` checkpoint. It cannot identify ships, land, sea
surface, or look-alikes as YOLO classes.

YOLO output is an investigation candidate only: it is not confirmed oil and not
a five-class classification. The existing Sentinel-1 flow is reused as CDSE
search/download -> SAFE calibration -> Lee filter -> filtered dB -> tiled YOLO
-> contour/vector output. YOLO receives its own uint8 rendering, never the
segmentation pathway's ImageNet-normalized tensor.

## Architecture and Rationale

Unlike the segmentation model which predicts pixel-exact class masks, YOLO is an object detector that outputs bounding boxes. To provide polygons for the frontend, the YOLO detector performs **contour extraction**:

1. Find bounding boxes using YOLO inference.
2. Crop the underlying filtered dB SAR data to the bounding box.
3. Apply Otsu thresholding, local quantile guards, and morphological cleanup to find the dark region (oil spill) inside the box.
4. Polygonize the largest/darkest connected component.

**Important Distinction:** The resulting polygon is *image-derived and approximate*. It is not a machine-learning segmentation mask. The `DetectionOutput` contract makes this explicit by tagging the geometry with `geometry_source="derived_contour"` and `geometry_quality="approximate"`. If contour extraction fails, it gracefully falls back to `geometry_source="bbox_fallback"`.

## Configuration

The YOLO MVP detector requires the `ultralytics` package (install with `uv sync --extra yolo`) and an external checkpoint file. It is configured via environment variables:

*   `OILSPILL_API_YOLO_WEIGHTS`: Absolute path to the YOLO checkpoint (e.g., `best.pt`). If unset, the YOLO pathway is gracefully disabled.
*   `OILSPILL_API_YOLO_CONF_THRESHOLD`: Inference confidence threshold (default: `0.25`).
*   `OILSPILL_API_YOLO_IOU_THRESHOLD`: NMS IoU threshold (default: `0.45`).
*   `OILSPILL_API_YOLO_TILE_SIZE`: Tiled inference size (default: `1024`).
*   `OILSPILL_API_YOLO_TILE_OVERLAP`: Tiled inference overlap (default: `128`).
*   `OILSPILL_API_YOLO_DB_MIN` / `OILSPILL_API_YOLO_DB_MAX`: The SAR rendering window (default: `-25.0` to `0.0`).

## Critical Validation Risk: SAR-to-YOLO Rendering

> **WARNING**: The exact SAR-to-image conversion used to train the YOLO model on Kaggle (`mmuthukumar07/oil-spill-dataset`) is undocumented.

The YOLO pipeline expects 8-bit, 3-channel RGB images. The `sar_to_yolo_image()` function renders these from the float32 dB SAR data using a linear window (`OILSPILL_API_YOLO_DB_MIN` to `OILSPILL_API_YOLO_DB_MAX`).

**This mapping is an explicit, unvalidated assumption.** Before deploying the YOLO MVP operationally, you **must** validate that this dB window matches the visual characteristics of the original Kaggle training images. If the mapping is wrong, the model's performance will degrade significantly.

## Heuristic Investigation Confidence

Because the YOLO model is a single-class detector trained only on "oil", its raw confidence does not reflect the likelihood of a look-alike (e.g., calm water, wind shadows).

To provide actionable UI feedback, the detector computes an **Investigation Confidence** score. This is a *heuristic ranking*, not a calibrated probability. It starts with the raw YOLO confidence and applies transparent penalties and bonuses:

*   **Land overlap**: Strong penalty (-0.4) for >50% overlap, moderate (-0.15) for >10%.
*   **Contour quality**: Minor penalties for failed extraction, tiny contours, or contours that fill the entire bounding box.
*   **Environmental context**: Penalty (-0.1) if wind speed is very low (<3 m/s), bonus (+0.05) if wind is moderate (>10 m/s). Bonus (+0.1) for optical corroboration.

The API returns the full breakdown of these adjustments in `investigation_confidence_breakdown` so users can understand exactly how the final score was derived. `model_confidence` is the raw YOLO output; `investigation_confidence` is a heuristic investigation score, not a calibrated probability.

An explicitly supplied optical conflict receives a penalty. Absent wind and optical values are never fetched or invented; they are returned as `wind_data_unavailable` and `optical_confirmation_unavailable` respectively.

## Output, GeoJSON, and Measurements

Tiled boxes are remapped to full-scene pixel coordinates, clipped at scene edges,
and merged using deterministic global NMS. Retained detections record all tile
indices that contributed a duplicate. Contour extraction uses the original
filtered dB crop, normalizing a local copy only for Otsu thresholding; it combines
a lower-quantile guard, conservative morphology, and connected components scored
by darkness, area fraction, distance to the box centre, and compactness.

On success, the selected component is polygonized using the scene affine transform,
repaired, simplified conservatively, and reprojected to EPSG:4326. It is labelled
`geometry_source="derived_contour"` and `geometry_quality="approximate"`. On a
uniform, invalid, empty, or poor crop, the API returns the georeferenced box with
`geometry_source="bbox_fallback"`, `geometry_quality="fallback"`, and
`contour_extraction_failed`. A fallback rectangle is an envelope, never a true
slick shape.

Every candidate carries raw/heuristic confidence, the complete score breakdown,
quality flags, envelope and derived-contour area, WGS84 centroid, geodesic
perimeter, major/minor axes, elongation, orientation, component count, and tile
provenance. Contour measurements are approximate. Geographic area/perimeter use
geodesic calculations; longitude/latitude degrees are not treated as planar metres.

## Full Configuration

All variables are prefixed `OILSPILL_API_`:

| Suffix | Default | Description |
| --- | --- | --- |
| `YOLO_WEIGHTS` | unset | External one-class `.pt` checkpoint. |
| `YOLO_CONF_THRESHOLD` | `0.25` | YOLO confidence threshold. |
| `YOLO_IOU_THRESHOLD` | `0.45` | Cross-tile global NMS IoU threshold. |
| `YOLO_TILE_SIZE` | `1024` | YOLO tile size. |
| `YOLO_TILE_OVERLAP` | `128` | Tile overlap in pixels. |
| `YOLO_DB_MIN` / `YOLO_DB_MAX` | `-25` / `0` | dB-to-uint8 window. |
| `YOLO_CONTOUR_MIN_PIXELS` | `50` | Minimum contour component pixels. |
| `YOLO_CONTOUR_MORPH_SIZE` | `3` | Morphology kernel side length. |
| `YOLO_CONTOUR_SIMPLIFY_TOLERANCE` | `1.0` | Simplification tolerance in projected scene units. |
| `YOLO_LOW_WIND_THRESHOLD` | `3.0` | Low-wind flag threshold in m/s. |
| `COASTLINES_PATH` | unset | Optional Natural Earth land geometry for the shared land mask. |

Install the optional dependency with `uv sync --extra yolo --group dev`, then set
`OILSPILL_API_YOLO_WEIGHTS` to the external checkpoint. `GET /yolo/status` and
YOLO job errors never expose the configured local filesystem path; scene-job
submission returns HTTP 503 when weights or `ultralytics` are unavailable.

## Limitations

The YOLO MVP pathway explicitly does **not** implement:
*   AIS ingestion or vessel correlation analysis.
*   Wind/current data ingestion (beyond accepting a static value in the API request).
*   Hindcasting or forward drift prediction.

Historical vessel reconstruction, wind/current retrieval, and suspect scoring are
also intentionally outside this MVP. Static environmental context is accepted only
when supplied by the API caller.

In the UI, these detections must always be treated as *candidates for investigation*, not confirmed oil spills.
