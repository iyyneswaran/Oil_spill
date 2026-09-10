import { useHealthz } from "../hooks/useHealthz";
import { useYoloStatus } from "../hooks/useYoloStatus";
import { useModels } from "../hooks/useModels";

/** Strip local filesystem paths from error messages as a defensive measure. */
function sanitize(text: string): string {
  return text.replace(/[A-Za-z]:\\[^\s'"]+/g, "[path]").replace(/\/[^\s'"]*\/[^\s'"]+/g, "[path]");
}

type Signal = "ok" | "warn" | "off" | "error";

function Dot({ state }: { state: Signal }) {
  return <span className={`dot ${state}`} />;
}

export default function ConsoleStrip() {
  const health = useHealthz();
  const yolo = useYoloStatus();
  const models = useModels();

  const apiState: Signal = health.isLoading
    ? "off"
    : health.isError
      ? "error"
      : "ok";

  const modelLoaded = models.data?.models.some((m) => m.available);
  const modelState: Signal = models.isLoading
    ? "off"
    : modelLoaded
      ? "ok"
      : "warn";

  const yoloState: Signal = yolo.isLoading
    ? "off"
    : yolo.data?.available
      ? "ok"
      : "warn";

  const yoloLabel = yolo.isLoading
    ? "YOLO checking"
    : yolo.data?.available
      ? "YOLO ready"
      : yolo.data
        ? sanitize(yolo.data.detail)
        : "YOLO unknown";

  return (
    <div className="console-strip" role="status" aria-label="System status">
      <div className="signal">
        <Dot state={apiState} />
        {apiState === "ok" ? "API online" : apiState === "error" ? "API offline" : "connecting"}
      </div>
      <div className="signal">
        <Dot state={modelState} />
        {modelState === "ok" ? "model loaded" : modelState === "warn" ? "no model" : "checking"}
      </div>
      <div className="signal">
        <Dot state={yoloState} />
        {yoloLabel}
      </div>
    </div>
  );
}
