import { useState, useRef, useEffect } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { runHindcast } from "../lib/api";
import type { HindcastResponse } from "../lib/types";

const RASTER_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export default function Hindcast() {
  const mapEl = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);

  const [lat, setLat] = useState<string>("-20.44");
  const [lon, setLon] = useState<string>("57.72");
  const [time, setTime] = useState<string>("2020-07-25T04:35:00Z");
  const [area, setArea] = useState<string>("3.1");
  const [duration, setDuration] = useState<string>("24");
  const [particles, setParticles] = useState<string>("250");
  
  // Historical forcing defaults
  const [currentU, setCurrentU] = useState<string>("0.10");
  const [currentV, setCurrentV] = useState<string>("-0.03");
  const [windU, setWindU] = useState<string>("3.0");
  const [windV, setWindV] = useState<string>("1.2");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<HindcastResponse | null>(null);

  useEffect(() => {
    if (!mapEl.current || map.current) return;
    const m = new maplibregl.Map({
      container: mapEl.current,
      style: RASTER_STYLE,
      center: [Number(lon) || 57.72, Number(lat) || -20.44],
      zoom: 8,
      attributionControl: false,
    });
    
    m.on("load", () => {
      m.addSource("hindcast", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "hindcast-lines",
        type: "line",
        source: "hindcast",
        filter: ["==", ["geometry-type"], "LineString"],
        paint: {
          "line-color": "#3498db",
          "line-width": 1.5,
          "line-opacity": 0.4,
        },
      });
      m.addLayer({
        id: "hindcast-points",
        type: "circle",
        source: "hindcast",
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-radius": 7,
          "circle-color": [
            "match",
            ["get", "result"],
            "current_oil_slick", "#e74c3c",
            "probable_spill_origin", "#f39c12",
            "#3498db",
          ],
          "circle-stroke-width": 2,
          "circle-stroke-color": "#ffffff",
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
    if (map.current && result?.geojson) {
      const m = map.current;
      if (!m.isStyleLoaded()) return;
      const src = m.getSource("hindcast") as maplibregl.GeoJSONSource | undefined;
      if (src) {
        src.setData(result.geojson as any);
        
        const coords = result.geojson.features
          .filter((f: any) => f.geometry.type === "Point")
          .map((f: any) => f.geometry.coordinates);
          
        if (coords.length > 0) {
          const bounds = coords.reduce((b: maplibregl.LngLatBounds, c: any) => {
            return b.extend(c as [number, number]);
          }, new maplibregl.LngLatBounds(coords[0], coords[0]));
          
          m.fitBounds(bounds, { padding: 80 });
        }
      }
    }
  }, [result]);

  const executeSimulation = async () => {
    setLoading(true);
    setError(null);

    const latNum = Number(lat);
    const lonNum = Number(lon);
    const areaNum = Number(area);
    const durNum = Math.max(1, Number(duration) || 24);
    const partCount = Math.max(10, Number(particles) || 250);

    const detDate = new Date(time);
    const validDate = isNaN(detDate.getTime()) ? new Date() : detDate;
    const startTimeStr = new Date(validDate.getTime() - (durNum + 2) * 3600 * 1000).toISOString();
    const endTimeStr = new Date(validDate.getTime() + 2 * 3600 * 1000).toISOString();

    const payload = {
      current_oil_slick: {
        latitude: latNum,
        longitude: lonNum,
        detection_time: validDate.toISOString(),
        slick_area_km2: areaNum,
      },
      ocean_currents: [
        { timestamp: startTimeStr, eastward_ms: Number(currentU) || 0.1, northward_ms: Number(currentV) || -0.03 },
        { timestamp: endTimeStr, eastward_ms: Number(currentU) || 0.1, northward_ms: Number(currentV) || -0.03 },
      ],
      winds: [
        { timestamp: startTimeStr, eastward_ms: Number(windU) || 3.0, northward_ms: Number(windV) || 1.2 },
        { timestamp: endTimeStr, eastward_ms: Number(windU) || 3.0, northward_ms: Number(windV) || 1.2 },
      ],
      config: {
        duration_hours: durNum,
        particle_count: partCount,
      },
    };

    try {
      const data = await runHindcast(payload);
      setResult(data);
    } catch (err: any) {
      setError(err?.message || "Failed to execute hindcast simulation");
    } finally {
      setLoading(false);
    }
  };

  const driftDist =
    result?.json?.probable_spill_origin && result?.json?.current_oil_slick_location
      ? haversineKm(
          result.json.current_oil_slick_location.latitude,
          result.json.current_oil_slick_location.longitude,
          result.json.probable_spill_origin.latitude,
          result.json.probable_spill_origin.longitude,
        )
      : null;

  return (
    <section>
      <div className="view-head">
        <h1>Backward Drift Simulation (Hindcasting)</h1>
        <p>
          Rewind an oil slick hour-by-hour using reverse Lagrangian particle tracking against
          historical winds and ocean currents to locate the probable spill origin.
        </p>
      </div>

      <div className="scene-grid">
        <div className="panel">
          <h3 style={{ marginBottom: "1rem" }}>Observed Slick Parameters</h3>
          <div className="field" style={{ marginBottom: "0.75rem" }}>
            <label>Latitude (°N)</label>
            <input type="number" step="0.0001" value={lat} onChange={(e) => setLat(e.target.value)} />
          </div>
          <div className="field" style={{ marginBottom: "0.75rem" }}>
            <label>Longitude (°E)</label>
            <input type="number" step="0.0001" value={lon} onChange={(e) => setLon(e.target.value)} />
          </div>
          <div className="field" style={{ marginBottom: "0.75rem" }}>
            <label>Detection Time (UTC ISO)</label>
            <input type="text" value={time} onChange={(e) => setTime(e.target.value)} />
          </div>
          <div className="field" style={{ marginBottom: "0.75rem" }}>
            <label>Slick Area (km²)</label>
            <input type="number" step="0.1" value={area} onChange={(e) => setArea(e.target.value)} />
          </div>

          <h3 style={{ marginTop: "1.25rem", marginBottom: "1rem" }}>Simulation Configuration</h3>
          <div className="row" style={{ marginBottom: "0.75rem" }}>
            <div className="field">
              <label>Duration (hours)</label>
              <input type="number" value={duration} onChange={(e) => setDuration(e.target.value)} />
            </div>
            <div className="field">
              <label>Particles</label>
              <input type="number" value={particles} onChange={(e) => setParticles(e.target.value)} />
            </div>
          </div>

          <h3 style={{ marginTop: "1.25rem", marginBottom: "1rem" }}>Historical Forcing Vectors</h3>
          <div className="row" style={{ marginBottom: "0.75rem" }}>
            <div className="field">
              <label>Current U (m/s, East)</label>
              <input type="number" step="0.01" value={currentU} onChange={(e) => setCurrentU(e.target.value)} />
            </div>
            <div className="field">
              <label>Current V (m/s, North)</label>
              <input type="number" step="0.01" value={currentV} onChange={(e) => setCurrentV(e.target.value)} />
            </div>
          </div>
          <div className="row" style={{ marginBottom: "1.25rem" }}>
            <div className="field">
              <label>Wind U (m/s, East)</label>
              <input type="number" step="0.1" value={windU} onChange={(e) => setWindU(e.target.value)} />
            </div>
            <div className="field">
              <label>Wind V (m/s, North)</label>
              <input type="number" step="0.1" value={windV} onChange={(e) => setWindV(e.target.value)} />
            </div>
          </div>

          <button
            className="primary"
            style={{ width: "100%" }}
            disabled={loading}
            onClick={executeSimulation}
          >
            {loading ? (
              <>
                <span className="spinner" /> Simulating Backward Drift…
              </>
            ) : (
              "Run Hindcast Simulation"
            )}
          </button>

          {error && (
            <div className="error-box" style={{ marginTop: "1rem" }}>
              {error}
            </div>
          )}

          {result && (
            <div style={{ marginTop: "1.5rem" }}>
              <h3>Simulation Results</h3>
              <div className="stat">
                <span>Origin Confidence</span>
                <span className="val">{Math.round(result.json.origin_probability * 100)}%</span>
              </div>
              {driftDist !== null && (
                <div className="stat">
                  <span>Net Drift Distance</span>
                  <span className="val">{driftDist.toFixed(2)} km</span>
                </div>
              )}
              <div className="stat">
                <span>Estimated Spill Origin</span>
                <span className="val">
                  {result.json.probable_spill_origin.latitude.toFixed(4)},{" "}
                  {result.json.probable_spill_origin.longitude.toFixed(4)}
                </span>
              </div>
              <div className="stat">
                <span>Estimated Spill Time</span>
                <span className="val" style={{ fontSize: "0.85rem" }}>
                  {result.json.estimated_spill_time}
                </span>
              </div>
              <div className="stat">
                <span>Particle Trajectories</span>
                <span className="val">{result.json.backward_trajectories.length}</span>
              </div>

              <div style={{ display: "flex", gap: "1rem", marginTop: "1rem", fontSize: "0.85rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#e74c3c", display: "inline-block" }} />
                  <span>Observed Slick</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#f39c12", display: "inline-block" }} />
                  <span>Probable Origin Zone</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ width: 16, height: 2, background: "#3498db", display: "inline-block" }} />
                  <span>Trajectories</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div ref={mapEl} className="map" />
      </div>
    </section>
  );
}

