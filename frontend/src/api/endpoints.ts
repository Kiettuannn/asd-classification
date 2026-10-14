/**
 * Centralized API endpoint dictionary.
 * All backend route paths are defined here and imported by feature hooks/api modules.
 * @see src/api/routes/ for corresponding FastAPI route handlers.
 */
export const ENDPOINTS = {
  /** GET /health – service status and loaded model info */
  HEALTH: '/health',
  /** POST /predict – classify subject FC matrix into ASD vs TD */
  PREDICT: '/predict',
  /** POST /explain – saliency / top salient connections for a prediction */
  EXPLAIN: '/explain',
} as const;

export type EndpointKey = keyof typeof ENDPOINTS;
