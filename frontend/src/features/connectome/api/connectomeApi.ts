/**
 * Connectome feature API module.
 * Retrieves explainability / saliency data for a given subject.
 */
import apiClient from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import type { PredictRequest, ExplainResponse } from '@/types/api';

/**
 * POST /explain – Retrieve the top salient connections for a subject.
 */
export async function fetchConnectomeExplain(payload: PredictRequest): Promise<ExplainResponse> {
  const { data } = await apiClient.post<ExplainResponse>(ENDPOINTS.EXPLAIN, payload);
  return data;
}
