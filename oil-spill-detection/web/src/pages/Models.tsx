import { useModels } from '../hooks/useModels';
import { CLASSES } from '../lib/classes';

function pct(v: number | undefined): string {
  if (v === undefined || Number.isNaN(v)) return '—';
  // Metrics may arrive as fractions (0–1) or percentages; normalise to %.
  const n = v <= 1 ? v * 100 : v;
  return `${n.toFixed(1)}`;
}

export default function Models() {
  const { data, isLoading, isError, error } = useModels();
  const models = data?.models ?? [];

  return (
    <section>
      <div className="view-head">
        <h1>Models</h1>
        <p>
          Honest per-class performance on the held-out test set.{' '}
          <strong style={{ color: 'var(--accent)' }}>
            Oil-class IoU is the headline metric
          </strong>{' '}
          — it measures how well the oil region is actually delineated. Pixel
          accuracy is reported for completeness but is misleading on imbalanced
          scenes where sea dominates, so treat it as secondary.
        </p>
      </div>

      {isLoading && (
        <p className="muted">
          Loading models…
        </p>
      )}
      {isError && (
        <div className="error-box">
          {error instanceof Error ? error.message : String(error)}
        </div>
      )}

      {!isLoading && !isError && (
        <>
          <div className="panel">
            <h2>Headline metrics (%)</h2>
            <div className="table-wrap">
              <table className="metrics" data-testid="models-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="headline" style={{ color: 'var(--accent)', fontSize: '1.2em' }}>Oil IoU</th>
                    <th className="headline">Oil recall</th>
                    <th>Mean IoU</th>
                    <th>Macro F1</th>
                    <th>Pixel acc.</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.id}>
                      <td>{m.name}</td>
                      <td className="headline" style={{ color: 'var(--accent)', fontSize: '1.2em', fontWeight: 'bold' }}>{pct(m.oil_iou)}</td>
                      <td className="headline" style={{ fontWeight: '600' }}>{pct(m.oil_recall)}</td>
                      <td>{pct(m.mean_iou)}</td>
                      <td>{pct(m.macro_f1)}</td>
                      <td style={{ opacity: 0.7 }}>{pct(m.pixel_accuracy)}</td>
                      <td>
                        {m.available ? (
                          <span className="badge ok">available</span>
                        ) : (
                          <span className="badge off">unavailable</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="metric-note">
              Pixel accuracy is secondary: a model that labels everything "sea"
              can still score high on a sea-dominated scene.
            </p>
          </div>

          <div className="panel perclass">
            <h2>Per-class breakdown (%)</h2>
            <div className="table-wrap">
              <table className="metrics">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Class</th>
                    <th>IoU</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  {models.flatMap((m) => {
                    const keys = Object.keys(m.per_class ?? {});
                    // Order by canonical class list, then any extras.
                    const ordered = [
                      ...CLASSES.map((c) => c.name).filter((n) =>
                        keys.includes(n),
                      ),
                      ...keys.filter(
                        (k) => !CLASSES.some((c) => c.name === k),
                      ),
                    ];
                    return ordered.map((cls, i) => {
                      const pc = m.per_class[cls];
                      const isOil = cls === 'Oil Spill';
                      return (
                        <tr key={`${m.id}-${cls}`}>
                          <td>{i === 0 ? m.name : ''}</td>
                          <td className={isOil ? 'headline' : ''}>{cls}</td>
                          <td className={isOil ? 'headline' : ''} style={isOil ? { color: 'var(--accent)', fontWeight: 'bold' } : {}}>
                            {pct(pc?.iou)}
                          </td>
                          <td>{pct(pc?.precision)}</td>
                          <td>{pct(pc?.recall)}</td>
                          <td>{pct(pc?.f1)}</td>
                        </tr>
                      );
                    });
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
