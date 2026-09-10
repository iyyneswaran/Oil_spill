import { useEffect, useMemo, useRef, useState } from 'react';
import { sampleSrc } from '../lib/api';
import Legend from '../components/Legend';
import type { Sample } from '../lib/types';
import { CLASSES } from '../lib/classes';
import { useModels } from '../hooks/useModels';
import { useSamples } from '../hooks/useSamples';
import { usePredict } from '../hooks/usePredict';

interface Selection {
  src: string;
  blob: Blob;
  filename: string;
}

export default function QuickDetect() {
  const modelsQuery = useModels();
  const samplesQuery = useSamples();
  const predict = usePredict();

  const models = useMemo(() => {
    if (!modelsQuery.data) return [];
    const available = modelsQuery.data.models.filter((m) => m.available);
    return available.length ? available : modelsQuery.data.models;
  }, [modelsQuery.data]);

  const samples = samplesQuery.data?.samples || [];

  const [modelId, setModelId] = useState<string>('');
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [opacity, setOpacity] = useState(0.65);
  const [dragOver, setDragOver] = useState(false);
  
  // Track visibility per class, initially all visible
  const [visibleClasses, setVisibleClasses] = useState<Set<string>>(
    new Set(Object.values(CLASSES).map((c) => c.name))
  );

  const fileInput = useRef<HTMLInputElement>(null);
  const objectUrl = useRef<string | null>(null);

  useEffect(() => {
    if (models.length > 0 && !modelId) {
      setModelId(models[0].id);
    }
  }, [models, modelId]);

  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  function setUploadedFile(file: File) {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    const url = URL.createObjectURL(file);
    objectUrl.current = url;
    setActiveSampleId(null);
    predict.reset();
    setSelection({ src: url, blob: file, filename: file.name });
  }

  async function selectSample(s: Sample) {
    setActiveSampleId(s.id);
    predict.reset();
    const src = sampleSrc(s.url);
    try {
      const res = await fetch(src);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const filename = s.url.split('/').pop() || `${s.id}.png`;
      setSelection({ src, blob, filename });
    } catch (e) {
      console.error('Could not load sample:', e);
    }
  }

  function runDetect() {
    if (!selection || !modelId) return;
    predict.mutate({ file: selection.blob, model: modelId, filename: selection.filename });
  }

  function downloadMask() {
    if (!predict.data) return;
    const a = document.createElement('a');
    a.href = predict.data.mask_png;
    a.download = 'oil-spill-mask.png';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  const toggleClass = (className: string) => {
    setVisibleClasses((prev) => {
      const next = new Set(prev);
      if (next.has(className)) {
        next.delete(className);
      } else {
        next.add(className);
      }
      return next;
    });
  };

  const canDetect = useMemo(
    () => !!selection && !!modelId && !predict.isPending,
    [selection, modelId, predict.isPending]
  );

  const selectedModel = models.find((m) => m.id === modelId);

  return (
    <section>
      <div className="view-head">
        <h1>Quick Detect</h1>
        <p>
          Run segmentation on a single image. Pick a built-in sample for an
          instant result, or drop in your own satellite scene.
        </p>
      </div>

      <div className="detect-grid">
        <div className="panel">
          <div className="field" style={{ marginBottom: '1rem' }}>
            <label htmlFor="model">Model</label>
            <select
              id="model"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              data-testid="model-select"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name}
                  {m.available ? '' : ' (unavailable)'}
                </option>
              ))}
            </select>
          </div>

          <div
            className={`dropzone${dragOver ? ' over' : ''}`}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = e.dataTransfer.files?.[0];
              if (f) setUploadedFile(f);
            }}
            data-testid="dropzone"
          >
            <strong>Drop an image</strong>
            <div style={{ fontSize: '0.85rem', opacity: 0.7 }}>
              or click to browse
            </div>
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) setUploadedFile(f);
              }}
            />
          </div>

          {samples.length > 0 && (
            <>
              <div
                style={{ fontSize: '0.85rem', marginTop: '1rem', opacity: 0.7 }}
              >
                Or pick a sample
              </div>
              <div className="samples">
                {samples.map((s) => (
                  <button
                    key={s.id}
                    className={activeSampleId === s.id ? 'active' : ''}
                    onClick={() => selectSample(s)}
                    data-testid={`sample-${s.id}`}
                    title={s.id}
                  >
                    <img src={sampleSrc(s.url)} alt={s.id} loading="lazy" />
                  </button>
                ))}
              </div>
            </>
          )}

          <button
            className="primary"
            style={{ width: '100%', marginTop: '1rem' }}
            disabled={!canDetect}
            onClick={runDetect}
            data-testid="detect-btn"
          >
            {predict.isPending ? (
              <>
                <span className="spinner" /> Running inference on {selectedModel?.name || 'model'}...
              </>
            ) : (
              'Detect'
            )}
          </button>
          
          <div style={{ fontSize: '0.85rem', marginTop: '0.75rem', opacity: 0.8 }}>
            Segmentation only. For YOLO candidate detection, use Scene Monitor with a full SAR scene.
          </div>

          {predict.isError && (
            <div className="error-box" style={{ marginTop: '1rem' }}>
              {predict.error instanceof Error ? predict.error.message : String(predict.error)}
            </div>
          )}
        </div>

        <div className="panel">
          {!selection ? (
            <p style={{ opacity: 0.7 }}>Select a sample image above or drop your own SAR scene to begin detection.</p>
          ) : (
            <>
              <div className="compare">
                <figure>
                  <figcaption>Original</figcaption>
                  <div className="image-frame">
                    <img src={selection.src} alt="Original input" />
                  </div>
                </figure>
                <figure>
                  <figcaption>Detection overlay</figcaption>
                  <div className="image-frame">
                    <img src={selection.src} alt="Base" />
                    {predict.data && (
                      <img
                        className="overlay"
                        src={predict.data.overlay_png}
                        alt="Segmentation overlay"
                        style={{ opacity }}
                        data-testid="overlay-img"
                      />
                    )}
                  </div>
                </figure>
              </div>

              {predict.data && (
                <>
                  <div className="field" style={{ marginTop: '1rem' }}>
                    <label htmlFor="opacity">
                      Overlay opacity — {Math.round(opacity * 100)}%
                    </label>
                    <input
                      id="opacity"
                      className="slider"
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={opacity}
                      onChange={(e) => setOpacity(Number(e.target.value))}
                      data-testid="opacity-slider"
                    />
                  </div>

                  <Legend
                    percentages={predict.data.class_percentages}
                    legend={predict.data.legend}
                    visibleClasses={visibleClasses}
                    onToggleClass={toggleClass}
                  />

                  <button
                    style={{ marginTop: '1rem' }}
                    onClick={downloadMask}
                    data-testid="download-mask"
                  >
                    Download mask (PNG)
                  </button>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
