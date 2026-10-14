/**
 * TypeScript interfaces mirroring the backend FastAPI Pydantic schemas.
 * @see src/api/schemas.py
 */

/** Health check response from GET /api/v1/health */
export interface HealthResponse {
  status: string;
  version: string;
  active_model?: string;
  atlas: string;
  num_rois: number;
}

/** Prediction request payload for POST /api/v1/predict */
export interface PredictRequest {
  /** Optional ABIDE subject identifier (e.g. "sub-0050003") */
  subject_id?: string;
  /** 200×200 Functional Connectivity Pearson correlation matrix */
  fc_matrix?: number[][];
}

/** Prediction response from POST /api/v1/predict */
export interface PredictResponse {
  subject_id?: string;
  /** Predicted diagnostic label */
  diagnosis: 'ASD' | 'TD';
  /** Probability of Autism Spectrum Disorder label [0, 1] */
  asd_probability: number;
  /** Probability of Typically Developing label [0, 1] */
  td_probability: number;
  /** Confidence score [0, 1] (max of asd/td probabilities) */
  confidence: number;
}

/** A single salient brain connection from the explainability response */
export interface ConnectionWeight {
  /** Source ROI index (0-based) */
  roi_source: number;
  /** Target ROI index (0-based) */
  roi_target: number;
  /** Saliency / feature importance weight */
  weight: number;
}

/** Explainability response from POST /api/v1/explain */
export interface ExplainResponse {
  subject_id?: string;
  /** Top contributing functional connections ranked by saliency */
  top_connections: ConnectionWeight[];
  /** Name of the model used for inference */
  model_name: string;
}
