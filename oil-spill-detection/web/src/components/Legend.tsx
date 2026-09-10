import { CLASSES, OIL_CLASS_NAME, colorForClass, rgbToCss } from "../lib/classes";
import type { RGB } from "../lib/types";

interface Props {
  percentages: Record<string, number>;
  legend?: Record<string, RGB>;
  /** Optional: which classes are currently visible (for toggling overlay layers). */
  visibleClasses?: Set<string>;
  /** Called when a class visibility is toggled. */
  onToggleClass?: (className: string) => void;
}

function orderedKeys(percentages: Record<string, number>): string[] {
  const known = CLASSES.map((c) => c.name).filter(
    (name) => name in percentages,
  );
  const extra = Object.keys(percentages).filter(
    (k) => !CLASSES.some((c) => c.name === k),
  );
  return [...known, ...extra];
}

export default function Legend({ percentages, legend, visibleClasses, onToggleClass }: Props) {
  const keys = orderedKeys(percentages);
  const hasToggle = visibleClasses !== undefined && onToggleClass !== undefined;

  return (
    <div className="legend" data-testid="legend" role="list" aria-label="Class legend">
      {keys.map((key) => {
        const pct = percentages[key] ?? 0;
        const isOil = key === OIL_CLASS_NAME;
        const isVisible = !hasToggle || visibleClasses!.has(key);
        return (
          <div
            key={key}
            className={`legend-row${isOil ? " oil" : ""}`}
            data-testid="legend-row"
            role="listitem"
            style={!isVisible ? { opacity: 0.4 } : undefined}
          >
            {hasToggle && (
              <input
                type="checkbox"
                checked={isVisible}
                onChange={() => onToggleClass!(key)}
                aria-label={`Toggle ${key} visibility`}
                style={{ accentColor: "var(--accent)" }}
              />
            )}
            <span
              className="legend-swatch"
              style={{ background: rgbToCss(colorForClass(key, legend)) }}
            />
            <span className="legend-name">{key}</span>
            <span className="legend-pct">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
