/** Tipos de la API. Fuente de verdad: docs/CONTRATOS.md (1.3.0), sección 4. */

export interface ModelClass {
  id: number;
  name: string;
  color: string;
  /** Texto de UI. */
  label: string;
}

export interface AppLimits {
  max_upload_bytes: number;
  max_video_seconds: number;
  min_side_px: number;
  formats: string[];
}

export interface AppConfig {
  public_base_url: string;
  limits: AppLimits;
  moods: string[];
  classes: ModelClass[];
}

export interface AnalyzedClass {
  id: number;
  name: string;
  pixel_pct: number;
  confidence: number;
}

export interface AnalyzeResponse {
  width: number;
  height: number;
  mask_png_base64: string;
  classes: AnalyzedClass[];
  inference_ms: { segmentation: number | null; depth: number | null; total: number | null };
  depth_png_base64: string | null;
  model_version: string;
  model_sha256: string;
  conf_threshold: number;
}

export type OutputFormat = 'mp4' | 'gif';
export type BrushMode = 'add' | 'only';
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed';
export type JobStage = 'queued' | 'segmenting' | 'depth' | 'rendering' | 'encoding' | 'done';

export interface AnimateRequest {
  file: File;
  mood: string;
  output_format: OutputFormat;
  /** Solo imágenes; en un video el servidor lo ignora. */
  duration_s?: number;
  mask_strokes?: string;
  brush_mode?: BrushMode;
  gallery_opt_in?: boolean;
  seed?: number;
}

export interface AnimateAccepted {
  job_id: string;
  status: JobStatus;
  queue_position: number;
  status_url: string;
}

export interface JobResult {
  url: string;
  format: OutputFormat;
  width: number;
  height: number;
  duration_s: number;
  fps: number;
  size_bytes: number;
}

/** Una etapa aún no medida vale null, nunca 0. */
export interface JobTimings {
  segmentation: number | null;
  depth: number | null;
  render: number | null;
  encode: number | null;
  total: number | null;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  stage: JobStage;
  progress: number;
  mood: string;
  seed: number;
  created_at: string;
  updated_at: string;
  result: JobResult | null;
  timings_ms: JobTimings | null;
  error: { code: string; message: string } | null;
}

export interface HealthResponse {
  status: 'ok' | 'loading';
  model_loaded: boolean;
  depth_loaded: boolean;
  model_version?: string;
  model_sha256?: string;
  depth_sha256?: string;
}

export interface GalleryItem {
  job_id: string;
  mood: string;
  result_url: string;
  created_at: string;
}

export interface GalleryResponse {
  items: GalleryItem[];
}

export interface ModelMetrics {
  /** Solo en el metrics.json del modelo de desarrollo. */
  dummy?: boolean;
  model_sha256: string;
  dataset_manifest_sha256: string;
  split: string;
  num_images: number;
  evaluated_at: string;
  miou: number | null;
  dice_macro: number | null;
  pixel_accuracy: number | null;
  per_class: Record<string, { iou: number | null; dice: number | null; support_pixels: number }>;
  confusion_matrix: number[][];
  classes: string[];
}

/** Versión que anuncia el modelo de desarrollo con pesos aleatorios (contrato 2.3). */
export const DUMMY_MODEL_VERSION = 'dummy-dev';
