import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useModels } from '../hooks/useModels';
import { useYoloStatus } from '../hooks/useYoloStatus';
import { useSubmitSceneJob, useJobStatus, useJobList } from '../hooks/useSceneJob';
import DetectorToggle from '../components/DetectorToggle';
import type { Job, DetectorType, YoloRawDetection } from "../lib/types";
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

  const [bbox, setBbox] = useState<BBox>(DEFAULT_BBOX);
  const [start, setStart] = useState("2023-06-01");
  const [end, setEnd] = useState("2023-06-30");
  const [modelId, setModelId] = useState<string>("");
  const [detector, setDetector] = useState<DetectorType>("segmentation");
  const [windSpeed, setWindSpeed] = useState<string>("");
  const [opticalConfirm, setOpticalConfirm] = useState(false);
  const [yoloCandidates, setYoloCandidates] = useState<YoloRawDetection[]>([]);
  const [bboxError, setBboxError] = useState<string | null>(null);

  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const { jobIds, addJob, removeJob } = useJobList();
  const submitJob = useSubmitSceneJob();
  const jobQuery = useJobStatus(activeJobId);
  
  const modelsQuery = useModels();
  const models = modelsQuery.data?.models ?? [];
  
  const yoloQuery = useYoloStatus();
  const yoloStatus = yoloQuery.data;

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
          "fill-opacity": 0.2,
        },
      });
      m.addLayer({
        id: "yolo-line-solid",
        type: "line",
        source: "yolo",
        paint: {
          "line-color": "#d35400",
          "line-width": 1.5,
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
    if (models.length > 0 && !modelId) {
      const first = models.find((m) => m.available) ?? models[0];
      if (first) setModelId(first.id);
    }
  }, [models, modelId]);

  useEffect(() => {
    if (yoloStatus && !yoloStatus.available && detector === "yolo_mvp") {
      setDetector("segmentation");
    }
  }, [yoloStatus, detector]);

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
    let candidates: YoloRawDetection[] = [];

    const job = jobQuery.data;

    if (job?.status === "done" && job.result) {
      if (job.result.yolo_result_image || detector === "yolo_mvp") {
        yoloFc = job.result.geojson;
        candidates = yoloFc.features.map(
          (f) => f.properties as unknown as YoloRawDetection
        );
      } else {
        oilFc = job.result.geojson;
      }
    }

    oilSrc.setData(oilFc as unknown as GeoJSON.FeatureCollection);
    yoloSrc.setData(yoloFc as unknown as GeoJSON.FeatureCollection);
    setYoloCandidates(candidates);
  }, [jobQuery.data, detector]);

  async function submit() {
    const poly = bboxToPolygon(bbox);
    if (!poly) {
      setBboxError("Invalid bounding box: west<east and south<north required.");
      return;
    }
    setBboxError(null);
    
    submitJob.mutate(
      {
        aoi: poly,
        start,
        end,
        detector,
        ...(detector === "segmentation" && modelId ? { model: modelId } : {}),
        ...(detector === "yolo_mvp"
          ? {
              environmental_context: {
                ...(windSpeed ? { wind_speed_ms: Number(windSpeed) } : {}),
                ...(opticalConfirm ? { optical_corroboration: true } : {}),
              },
            }
          : {}),
      },
      {
        onSuccess: (data) => {
          addJob(data.job_id);
          setActiveJobId(data.job_id);
        }
      }
    );
  }

  function downloadGeoJSON() {
    const job = jobQuery.data;
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

  const job = jobQuery.data;
  const running =
    job?.status === "queued" || job?.status === "running" || submitJob.isPending;

  const is404 = jobQuery.isError && jobQuery.error?.message?.startsWith("404");

  // Handle 404 (server restarted) — clear the stale job from the list.
  useEffect(() => {
    if (is404 && activeJobId) {
      removeJob(activeJobId);
      setActiveJobId(null);
    }
  }, [is404, activeJobId, removeJob]);

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
        <div className="panel">
          <DetectorToggle
            value={detector}
            onChange={setDetector}
            yoloAvailable={yoloStatus?.available ?? null}
            yoloDetail={yoloStatus?.detail}
          />

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
            {bboxError && <div className="error-box" style={{ marginTop: "0.5rem" }}>{bboxError}</div>}
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
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
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

          {submitJob.error && !is404 && (
            <div className="error-box" style={{ marginTop: "1rem" }}>
              {submitJob.error.message}
            </div>
          )}

          {jobIds.length > 0 && (
            <div className="job-list" style={{ marginTop: "2rem" }}>
              <h3>Recent Jobs</h3>
              {jobIds.map((id) => (
                <div 
                  key={id} 
                  className={`job-item ${id === activeJobId ? 'active' : ''}`}
                  onClick={() => setActiveJobId(id)}
                  style={{ cursor: 'pointer', padding: '0.5rem', border: '1px solid var(--border)', marginBottom: '0.5rem', background: id === activeJobId ? 'var(--bg)' : 'transparent' }}
                >
                  <div style={{ fontSize: '0.85rem' }}>{id}</div>
                </div>
              ))}
            </div>
          )}
          
          {is404 && (
            <div className="error-box" style={{ marginTop: "1rem" }}>
              This job is no longer tracked. The server may have restarted. Submit a new one.
            </div>
          )}

          {job && !is404 && (
            <div style={{ marginTop: "2rem" }}>
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
                <div className="error-box" style={{ marginTop: "0.5rem" }}>
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
                        <strong>Investigation candidates — not confirmed spills.</strong>
                        <p style={{ margin: "0.25rem 0 0 0" }}>
                          The YOLO pathway flags bounding boxes worth investigating. It is not a confirmed detection.
                        </p>
                      </div>
                      
                      {job.result.yolo_result_image && (
                        <div style={{ marginTop: "1rem" }}>
                          <img 
                            src={job.result.yolo_result_image} 
                            alt="YOLO Detections" 
                            style={{ width: "100%" }} 
                          />
                        </div>
                      )}

                      {yoloCandidates.length === 0 ? (
                        <p style={{ marginTop: "1rem" }}>No candidates detected.</p>
                      ) : (
                        <div style={{ marginTop: "1rem", overflowX: "auto" }}>
                          <table style={{ width: "100%", textAlign: "left", borderCollapse: "collapse", fontSize: "var(--font-size-meta)", fontFamily: "var(--font-mono)" }}>
                            <thead>
                              <tr>
                                <th style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>Spill ID</th>
                                <th style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>Conf</th>
                                <th style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>Location</th>
                                <th style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>Class</th>
                              </tr>
                            </thead>
                            <tbody>
                              {yoloCandidates.map((c, i) => (
                                <tr key={i}>
                                  <td style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>{c.spill_id}</td>
                                  <td style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>
                                    {Math.round(c.confidence * 100)}%
                                  </td>
                                  <td style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>
                                    {c.latitude.toFixed(4)}, {c.longitude.toFixed(4)}
                                  </td>
                                  <td style={{ borderBottom: "1px solid var(--border)", padding: "4px" }}>{c.class_name}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
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
