import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { createSceneJob, getJob, getModels } from "../lib/api";
import type { Job, ModelInfo, DetectorType, YoloCandidateResult } from "../lib/types";
import type { FeatureCollection, Polygon } from "../lib/geojson";

// Key-free OpenStreetMap raster style. Tiles require visible attribution.
const RASTER_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

interface BBox {
  west: string;
  south: string;
  east: string;
  north: string;
}

const DEFAULT_BBOX: BBox = {
  west: "1.8",
  south: "40.9",
  east: "2.6",
  north: "41.4",
};

function bboxToPolygon(b: BBox): Polygon | null {
  const w = Number(b.west),
    s = Number(b.south),
    e = Number(b.east),
    n = Number(b.north);
  if ([w, s, e, n].some((v) => Number.isNaN(v))) return null;
  if (w >= e || s >= n) return null;
  return {
    type: "Polygon",
    coordinates: [
      [
        [w, s],
        [e, s],
        [e, n],
        [w, n],
        [w, s],
      ],
    ],
  };
}

const TERMINAL: Job["status"][] = ["done", "error"];

export default function SceneMonitor() {
  const mapEl = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [bbox, setBbox] = useState<BBox>(DEFAULT_BBOX);
  const [start, setStart] = useState("2023-06-01");
  const [end, setEnd] = useState("2023-06-30");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [detector, setDetector] = useState<DetectorType>("segmentation");
  const [windSpeed, setWindSpeed] = useState<string>("");
  const [opticalConfirm, setOpticalConfirm] = useState(false);
  const [yoloCandidates, setYoloCandidates] = useState<YoloCandidateResult[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Initialise the map once.
  useEffect(() => {
    if (!mapEl.current || map.current) return;
    const m = new maplibregl.Map({
      container: mapEl.current,
      style: RASTER_STYLE,
      center: [2.2, 41.15],
      zoom: 8,
      attributionControl: false,
    });
    m.addControl(new maplibregl.AttributionControl({ compact: false }));
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }));
    m.on("load", () => {
      m.addSource("aoi", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "aoi-fill",
        type: "fill",
        source: "aoi",
        paint: { "fill-color": "#4f9da6", "fill-opacity": 0.12 },
      });
      m.addLayer({
        id: "aoi-line",
        type: "line",
        source: "aoi",
        paint: { "line-color": "#5fb3bd", "line-width": 1.5 },
      });
      m.addSource("oil", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "oil-fill",
        type: "fill",
        source: "oil",
        paint: { "fill-color": "#00e5e5", "fill-opacity": 0.35 },
      });
      m.addLayer({
        id: "oil-line",
        type: "line",
        source: "oil",
        paint: { "line-color": "#00e5e5", "line-width": 1.5 },
      });
      m.addSource("yolo", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "yolo-fill",
        type: "fill",
        source: "yolo",
        paint: {
          "fill-color": "#f39c12",
          "fill-opacity": [
            "case",
            ["==", ["get", "geometry_source"], "derived_contour"],
            0.35,
            0.1
          ],
        },
      });
      m.addLayer({
        id: "yolo-line",
        type: "line",
        source: "yolo",
        paint: {
          "line-color": "#d35400",
          "line-width": 1.5,
          "line-dasharray": [
            "case",
            ["==", ["get", "geometry_source"], "bbox_fallback"],
            ["literal", [3, 3]],
            ["literal", [1, 0]]
          ],
        },
      });
    });
    map.current = m;
    return () => {
      m.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    getModels()
      .then((r) => {
        setModels(r.models);
        const first = r.models.find((m) => m.available) ?? r.models[0];
        if (first) setModelId(first.id);
      })
      .catch(() => {
        /* model is optional for scene jobs */
      });
  }, []);

  // Keep the AOI rectangle in sync with the bbox inputs.
  useEffect(() => {
    const m = map.current;
    const poly = bboxToPolygon(bbox);
    if (!m || !m.isStyleLoaded()) return;
    const src = m.getSource("aoi") as maplibregl.GeoJSONSource | undefined;
    if (!src) return;
    src.setData(
      poly
        ? {
            type: "FeatureCollection",
            features: [{ type: "Feature", geometry: poly, properties: {} }],
          }
        : { type: "FeatureCollection", features: [] },
    );
  }, [bbox]);

  // Render oil polygons when a job completes.
  useEffect(() => {
    const m = map.current;
    if (!m || !m.isStyleLoaded()) return;
    const oilSrc = m.getSource("oil") as maplibregl.GeoJSONSource | undefined;
    const yoloSrc = m.getSource("yolo") as maplibregl.GeoJSONSource | undefined;
    if (!oilSrc || !yoloSrc) return;

    let oilFc: FeatureCollection = { type: "FeatureCollection", features: [] };
    let yoloFc: FeatureCollection = { type: "FeatureCollection", features: [] };
    let candidates: YoloCandidateResult[] = [];

    if (job?.status === "done" && job.result) {
      if (detector === "yolo_mvp") {
        yoloFc = job.result.geojson;
        candidates = yoloFc.features.map(
          (f) => f.properties as unknown as YoloCandidateResult
        );
      } else {
        oilFc = job.result.geojson;
      }
    }

    oilSrc.setData(oilFc as unknown as GeoJSON.FeatureCollection);
    yoloSrc.setData(yoloFc as unknown as GeoJSON.FeatureCollection);
    setYoloCandidates(candidates);
  }, [job, detector]);

  useEffect(() => {
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, []);

  function poll(id: string) {
    getJob(id)
      .then((j) => {
        setJob(j);
        if (!TERMINAL.includes(j.status)) {
          pollTimer.current = setTimeout(() => poll(id), 2000);
        }
      })
      .catch((e) => setError(String((e as Error).message ?? e)));
  }

  async function submit() {
    const poly = bboxToPolygon(bbox);
    if (!poly) {
      setError("Invalid bounding box: west<east and south<north required.");
      return;
    }
    setError(null);
    setJob(null);
    setSubmitting(true);
    if (pollTimer.current) clearTimeout(pollTimer.current);
    try {
      const { job_id } = await createSceneJob({
        aoi: poly,
        start,
        end,
        detector,
        ...(detector === "segmentation" && modelId ? { model: modelId } : {}),
        ...(detector === "yolo_mvp"
          ? {
              environmental_context: {
                ...(windSpeed ? { wind_speed_ms: Number(windSpeed) } : {}),
                optical_corroboration: opticalConfirm,
              },
            }
          : {}),
      });
      poll(job_id);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setSubmitting(false);
    }
  }

  function downloadGeoJSON() {
    if (!job?.result) return;
    const blob = new Blob([JSON.stringify(job.result.geojson, null, 2)], {
      type: "application/geo+json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "oil-polygons.geojson";
    a.click();
    URL.revokeObjectURL(url);
  }

  const running =
    job?.status === "queued" || job?.status === "running" || submitting;

  return (
    <section>
      <div className="view-head">
        <h1>Scene Monitor</h1>
        <p>
          Define an area of interest and a date range, then queue a scene
          analysis. Detected oil polygons are drawn on the map with their total
          area.
        </p>
      </div>

      <div className="scene-grid">
        <div className="card">
          <div className="detector-select">
            <button
              className={detector === "segmentation" ? "active" : ""}
              onClick={() => setDetector("segmentation")}
            >
              Segmentation
            </button>
            <button
              className={detector === "yolo_mvp" ? "active" : ""}
              onClick={() => setDetector("yolo_mvp")}
            >
              YOLO MVP
            </button>
          </div>

          <div className="field" style={{ marginBottom: "1rem" }}>
            <label>Area of interest (bounding box, WGS84)</label>
            <div className="bbox-grid">
              <input
                type="number"
                step="0.01"
                placeholder="North"
                value={bbox.north}
                onChange={(e) => setBbox({ ...bbox, north: e.target.value })}
                aria-label="North latitude"
              />
              <input
                type="number"
                step="0.01"
                placeholder="East"
                value={bbox.east}
                onChange={(e) => setBbox({ ...bbox, east: e.target.value })}
                aria-label="East longitude"
              />
              <input
                type="number"
                step="0.01"
                placeholder="South"
                value={bbox.south}
                onChange={(e) => setBbox({ ...bbox, south: e.target.value })}
                aria-label="South latitude"
              />
              <input
                type="number"
                step="0.01"
                placeholder="West"
                value={bbox.west}
                onChange={(e) => setBbox({ ...bbox, west: e.target.value })}
                aria-label="West longitude"
              />
            </div>
          </div>

          <div className="row" style={{ marginBottom: "1rem" }}>
            <div className="field">
              <label htmlFor="start">Start</label>
              <input
                id="start"
                type="date"
                value={start}
                onChange={(e) => setStart(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="end">End</label>
              <input
                id="end"
                type="date"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
              />
            </div>
          </div>

          {detector === "segmentation" && models.length > 0 && (
            <div className="field" style={{ marginBottom: "1rem" }}>
              <label htmlFor="scene-model">Model</label>
              <select
                id="scene-model"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name}
                    {m.available ? "" : " (unavailable)"}
                  </option>
                ))}
              </select>
            </div>
          )}

          {detector === "yolo_mvp" && (
            <div className="env-context" style={{ marginBottom: "1rem" }}>
              <div className="field">
                <label>Wind speed (m/s)</label>
                <input
                  type="number"
                  step="0.1"
                  value={windSpeed}
                  onChange={(e) => setWindSpeed(e.target.value)}
                  placeholder="e.g. 5.2"
                />
              </div>
              <div className="field" style={{ marginTop: "0.5rem" }}>
                <label>
                  <input
                    type="checkbox"
                    checked={opticalConfirm}
                    onChange={(e) => setOpticalConfirm(e.target.checked)}
                  />
                  Optical confirmation (visible slicks)
                </label>
              </div>
            </div>
          )}

          <button
            className="primary"
            style={{ width: "100%" }}
            disabled={running}
            onClick={submit}
          >
            {running ? (
              <>
                <span className="spinner" /> Working…
              </>
            ) : (
              "Analyze scene"
            )}
          </button>

          {error && (
            <div className="error" style={{ marginTop: "1rem" }}>
              {error}
            </div>
          )}

          {job && (
            <div style={{ marginTop: "1rem" }}>
              <div className="status-line">
                <span className="muted">Status:</span>
                <strong>{job.status}</strong>
                {!TERMINAL.includes(job.status) && <span className="spinner" />}
              </div>
              {job.detail && (
                <p className="faint" style={{ fontSize: "0.85rem" }}>
                  {job.detail}
                </p>
              )}
              {job.status === "error" && (
                <div className="error" style={{ marginTop: "0.5rem" }}>
                  {job.detail ?? "The scene job failed."}
                </div>
              )}
              {job.status === "done" && job.result && (
                <>
                  {detector === "segmentation" ? (
                    <>
                      <div className="stat">
                        <span>Oil polygons</span>
                        <span className="val">{job.result.num_oil_polygons}</span>
                      </div>
                      <div className="stat">
                        <span>Total oil area</span>
                        <span className="val">
                          {job.result.total_oil_area_km2.toFixed(2)} km²
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="yolo-disclaimer">
                        <strong>YOLO MVP Detector</strong>
                        <p style={{ margin: "0.25rem 0 0 0" }}>
                          These are heuristic-based candidate detections. They require manual review.
                        </p>
                      </div>
                      {yoloCandidates.length === 0 ? (
                        <p style={{ marginTop: "1rem" }}>No candidates detected.</p>
                      ) : (
                        yoloCandidates.map((c, i) => (
                          <div key={i} className="candidate-card">
                            <h4>
                              Candidate {i + 1}
                              <span
                                className={`geometry-badge ${
                                  c.geometry_source === "derived_contour"
                                    ? "contour"
                                    : "fallback"
                                }`}
                              >
                                {c.geometry_source === "derived_contour"
                                  ? "Contour"
                                  : "BBox"}
                              </span>
                            </h4>
                            <div className="confidence-bars">
                              <div className="confidence-bar">
                                <span className="bar-label">Model Conf</span>
                                <div className="bar-track">
                                  <div
                                    className="bar-fill model"
                                    style={{ width: `${c.model_confidence * 100}%` }}
                                  />
                                </div>
                                <span className="bar-value">
                                  {Math.round(c.model_confidence * 100)}%
                                </span>
                              </div>
                              <div className="confidence-bar">
                                <span className="bar-label">Investigate Conf</span>
                                <div className="bar-track">
                                  <div
                                    className="bar-fill heuristic"
                                    style={{
                                      width: `${Math.max(0, Math.min(1, c.investigation_confidence)) * 100}%`,
                                    }}
                                  />
                                </div>
                                <span className="bar-value">
                                  {Math.round(c.investigation_confidence * 100)}%
                                </span>
                              </div>
                            </div>
                            
                            {c.investigation_confidence_breakdown?.length > 0 && (
                              <div className="breakdown">
                                {c.investigation_confidence_breakdown.map((b, j) => (
                                  <div key={j} className="breakdown-item">
                                    <span>{b.reason}</span>
                                    <span
                                      className={`breakdown-delta ${
                                        b.delta > 0 ? "positive" : "negative"
                                      }`}
                                    >
                                      {b.delta > 0 ? "+" : ""}
                                      {Math.round(b.delta * 100)}%
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}

                            <div className="candidate-measurements">
                              <div className="label">Area</div>
                              <div className="value">
                                {(
                                  c.derived_contour_area_km2 ?? c.candidate_envelope_area_km2
                                ).toFixed(2)}{" "}
                                km²
                              </div>
                              <div className="label">Elongation</div>
                              <div className="value">{c.elongation.toFixed(2)}</div>
                            </div>

                            {c.quality_flags?.length > 0 && (
                              <div className="quality-flags">
                                {c.quality_flags.map((f, j) => (
                                  <span key={j} className="quality-flag">
                                    {f}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </>
                  )}
                  <button
                    style={{ marginTop: "1rem", width: "100%" }}
                    onClick={downloadGeoJSON}
                  >
                    Download GeoJSON
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        <div ref={mapEl} className="map" />
      </div>
    </section>
  );
}
