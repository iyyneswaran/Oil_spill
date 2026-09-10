import { useSamples } from "../hooks/useSamples";
import { sampleSrc } from "../lib/api";
import PipelineDiagram from "../components/PipelineDiagram";
import { CLASSES, rgbToCss } from "../lib/classes";

export default function Overview() {
  const samplesQuery = useSamples();
  const samples = samplesQuery.data?.samples ?? [];
  const heroSample = samples.length > 0 ? samples[0] : null;

  return (
    <div>
      <div className="view-head">
        <h1>Overview</h1>
        <p>
          Sentinel-1 SAR oil-spill detection and forensic characterization,
          from raw satellite scenes to vectorized spill polygons.
        </p>
      </div>

      <div className="overview-hero">
        <div className="hero-image">
          {samplesQuery.isLoading && (
            <div style={{ padding: "var(--space-xl)", color: "var(--ink-muted)" }}>
              <span className="spinner" /> Loading sample image...
            </div>
          )}
          {samplesQuery.isError && (
            <div style={{ padding: "var(--space-xl)", color: "var(--danger)" }}>
              Could not load sample images from the API.
            </div>
          )}
          {heroSample && (
            <img
              src={sampleSrc(heroSample.url)}
              alt={`SAR sample: ${heroSample.id}`}
            />
          )}
        </div>

        <div>
          <h3>Class legend</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {CLASSES.map((cls) => (
              <div
                key={cls.id}
                style={{ display: "flex", alignItems: "center", gap: "8px" }}
              >
                <span
                  className="legend-swatch"
                  style={{ background: rgbToCss(cls.color) }}
                />
                <span className="mono">{cls.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ marginTop: "var(--space-2xl)" }}>
        <h2>Detection pipeline</h2>
        <PipelineDiagram />
      </div>

      <div className="distinction-block">
        <h3>How detection works</h3>

        <p style={{ marginTop: "var(--space-md)" }}>
          <strong>Segmentation</strong> examines every pixel of a SAR scene and
          classifies it into one of five categories: sea surface, oil spill,
          look-alike, ship, or land. The result is a pixel-exact mask — you can
          see precisely where oil begins and ends. All metrics on the Models
          page measure this pathway.
        </p>

        <p style={{ marginTop: "var(--space-md)" }}>
          <strong>YOLO MVP</strong> is a fast candidate screener. It draws
          bounding boxes around regions that may be oil, producing investigation
          candidates — not confirmed spills. It cannot distinguish ships, land,
          or look-alikes. Use it in Scene Monitor for rapid initial triage of
          large areas.
        </p>
      </div>
    </div>
  );
}
