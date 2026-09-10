import type { DetectorType } from "../lib/types";

interface Props {
  value: DetectorType;
  onChange: (detector: DetectorType) => void;
  yoloAvailable: boolean | null;
  yoloDetail?: string;
}

export default function DetectorToggle({ value, onChange, yoloAvailable, yoloDetail }: Props) {
  return (
    <div className="detector-toggle" role="group" aria-label="Detector pathway">
      <button
        type="button"
        className={value === "segmentation" ? "active" : ""}
        onClick={() => onChange("segmentation")}
        aria-pressed={value === "segmentation"}
      >
        Segmentation
      </button>
      <button
        type="button"
        className={value === "yolo_mvp" ? "active" : ""}
        disabled={yoloAvailable !== true}
        onClick={() => onChange("yolo_mvp")}
        aria-pressed={value === "yolo_mvp"}
        title={
          yoloAvailable === false
            ? yoloDetail || "YOLO detector is not available"
            : yoloAvailable === null
              ? "Checking YOLO availability..."
              : "YOLO MVP one-class oil candidate detector"
        }
      >
        YOLO MVP
        {yoloAvailable === false && " (unavailable)"}
      </button>
    </div>
  );
}
