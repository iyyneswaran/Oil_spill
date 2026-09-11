// Shapes mirror the backend API contract exactly.
import type { FeatureCollection, Polygon, Geometry } from "./geojson";

export interface PerClassMetric {
  iou: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  oil_iou: number;
  oil_recall: number;
  mean_iou: number;
  macro_f1: number;
  pixel_accuracy: number;
  per_class: Record<string, PerClassMetric>;
  available: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface Sample {
  id: string;
  url: string;
}

export interface SamplesResponse {
  samples: Sample[];
}

export type RGB = [number, number, number];

export interface PredictResponse {
  model: string;
  width: number;
  height: number;
  class_percentages: Record<string, number>;
  legend: Record<string, RGB>;
  mask_png: string;
  overlay_png: string;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface JobResult {
  num_oil_polygons: number;
  total_oil_area_km2: number;
  geojson: FeatureCollection;
  yolo_result_image?: string;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  detail?: string;
  result?: JobResult;
}

export interface SceneJobRequest {
  aoi: Polygon;
  start: string;
  end: string;
  model?: string;
  detector?: DetectorType;
  environmental_context?: EnvironmentalContext;
}

// --- YOLO MVP types --------------------------------------------------------

export interface ConfidenceAdjustment {
  reason: string;
  delta: number;
}

export type DetectorType = "segmentation" | "yolo_mvp";
export type GeometrySourceType = "derived_contour" | "bbox_fallback" | "segmentation_mask";
export type GeometryQualityType = "approximate" | "fallback" | "high";

export interface EnvironmentalContext {
  wind_speed_ms?: number;
  optical_corroboration?: boolean;
  optical_conflict?: boolean;
}

export interface YoloRawDetection {
  spill_id: string;
  latitude: number;
  longitude: number;
  bbox: [number, number, number, number];
  confidence: number;
  model_confidence: number;
  class_id: number;
  class_name: string;
  detector_type: DetectorType;
  model_id: string;
  tile_provenance: number[];
}

export interface YoloCandidateResult {
  detector_type: DetectorType;
  model_id: string;
  model_confidence: number;
  investigation_confidence: number;
  investigation_confidence_breakdown: ConfidenceAdjustment[];
  geometry_source: GeometrySourceType;
  geometry_quality: GeometryQualityType;
  quality_flags: string[];
  geometry: Geometry;
  candidate_envelope_area_km2: number;
  derived_contour_area_km2: number | null;
  centroid: [number, number];
  perimeter_km: number;
  major_axis_km: number;
  minor_axis_km: number;
  elongation: number;
  orientation_degrees: number;
  component_count: number | null;
  tile_provenance: number[];
}

export interface YoloDetectionResult {
  detector_type: DetectorType;
  model_id: string;
  scene_id: string;
  crs: string;
  bbox: number[];
  candidates: YoloCandidateResult[];
  scene_metadata: Record<string, unknown>;
}

export interface YoloStatusResponse {
  available: boolean;
  model_id: string | null;
  detail: string;
}

// --- Hindcast types --------------------------------------------------------

export interface HistoricalVector {
  timestamp: string;
  eastward_ms: number;
  northward_ms: number;
}

export interface CurrentOilSlickInput {
  latitude: number;
  longitude: number;
  detection_time: string;
  slick_area_km2: number;
}

export interface HindcastConfigInput {
  duration_hours?: number;
  step_minutes?: number;
  particle_count?: number;
  windage?: number;
  horizontal_diffusivity_m2_s?: number;
  random_seed?: number;
  uncertainty_confidence?: number;
}

export interface HindcastRequest {
  current_oil_slick: CurrentOilSlickInput;
  ocean_currents: HistoricalVector[];
  winds: HistoricalVector[];
  config?: HindcastConfigInput;
}

export interface HindcastResultData {
  current_oil_slick_location: {
    latitude: number;
    longitude: number;
    slick_area_km2: number;
  };
  detection_time: string;
  backward_trajectories: Array<{
    particle_id: number;
    coordinates: [number, number][];
  }>;
  probable_spill_origin: {
    latitude: number;
    longitude: number;
  };
  estimated_spill_time: string;
  origin_probability: number;
}

export interface HindcastResponse {
  json: HindcastResultData;
  geojson: FeatureCollection;
}
