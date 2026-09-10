import { Link } from "react-router-dom";

interface Stage {
  num: string;
  name: string;
  desc: string;
  live: boolean;
  link?: string;
}

const STAGES: Stage[] = [
  { num: "01", name: "Intake", desc: "Sentinel-1 SAR scene search and download via CDSE", live: true, link: "/scenes" },
  { num: "02", name: "Detect", desc: "5-class segmentation or YOLO candidate detection", live: true, link: "/detect" },
  { num: "03", name: "Characterize", desc: "Oil polygon vectorization, area and shape metrics", live: true, link: "/scenes" },
  { num: "04", name: "Hindcast", desc: "Reverse drift modeling to find probable spill origin", live: false },
  { num: "05", name: "AIS match", desc: "Correlate vessel tracks with spill location and timing", live: false },
  { num: "06", name: "Suspect score", desc: "Rank nearby vessels by proximity and behaviour", live: false },
  { num: "07", name: "Forecast", desc: "Forward drift prediction for response planning", live: false },
  { num: "08", name: "Dashboard", desc: "Investigation case management and reporting", live: false },
];

export default function PipelineDiagram() {
  return (
    <div className="pipeline" role="list" aria-label="Detection pipeline stages">
      {STAGES.map((s) => (
        <div
          key={s.num}
          className={`pipeline-stage ${s.live ? "live" : "roadmap"}`}
          role="listitem"
        >
          <div className="stage-num">{s.num}</div>
          <div className="stage-name">
            {s.live && s.link ? (
              <Link to={s.link}>{s.name}</Link>
            ) : (
              s.name
            )}
          </div>
          <p className="stage-desc">{s.desc}</p>
          <div className="stage-marker">
            {s.live ? "◆ live" : "○ roadmap"}
          </div>
        </div>
      ))}
    </div>
  );
}
