/**
 * Predict feature API module.
 * Wraps the Axios client for prediction and explanation endpoints.
 */
import apiClient from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import type { PredictRequest, PredictResponse, ExplainResponse } from '@/types/api';

/**
 * POST /predict – Classify subject FC matrix into ASD vs TD.
 */
export async function postPredict(payload: PredictRequest): Promise<PredictResponse> {
  const { data } = await apiClient.post<PredictResponse>(ENDPOINTS.PREDICT, payload);
  return data;
}

/**
 * POST /explain – Retrieve saliency / top connections for a subject.
 */
export async function postExplain(payload: PredictRequest): Promise<ExplainResponse> {
  const { data } = await apiClient.post<ExplainResponse>(ENDPOINTS.EXPLAIN, payload);
  return data;
}
