/**
 * MSW request handlers for testing.
 * Mocks the ASD Classification backend API at the HTTP boundary.
 */
import { http, HttpResponse } from 'msw';
import type { PredictResponse, ExplainResponse, HealthResponse } from '@/types/api';
import type { CohortSummary, ModelBenchmark } from '@/types/cohort';

const BASE = 'http://localhost:8000/api/v1';

export const handlers = [
  // GET /health
  http.get(`${BASE}/health`, () => {
    const response: HealthResponse = {
      status: 'ok',
      version: '0.1.0',
      atlas: 'cc200',
      num_rois: 200,
    };
    return HttpResponse.json(response);
  }),

  // POST /predict
  http.post(`${BASE}/predict`, () => {
    const response: PredictResponse = {
      subject_id: 'sub-0050003',
      diagnosis: 'ASD',
      asd_probability: 0.82,
      td_probability: 0.18,
      confidence: 0.82,
    };
    return HttpResponse.json(response);
  }),

  // POST /explain
  http.post(`${BASE}/explain`, () => {
    const response: ExplainResponse = {
      subject_id: 'sub-0050003',
      top_connections: [
        { roi_source: 12, roi_target: 45, weight: 0.934 },
        { roi_source: 7,  roi_target: 88, weight: 0.812 },
        { roi_source: 31, roi_target: 102, weight: 0.778 },
      ],
      model_name: 'GCN-v1',
    };
    return HttpResponse.json(response);
  }),

  // Deferred cohort endpoints used by the scaffold's query hooks.
  http.get(`${BASE}/cohort/summary`, () => {
    const response: CohortSummary = {
      total_subjects: 1_112,
      asd_count: 539,
      td_count: 573,
      mean_age: 16.8,
      sites: [],
    };
    return HttpResponse.json(response);
  }),

  http.get(`${BASE}/cohort/benchmarks`, () => {
    const response: ModelBenchmark[] = [];
    return HttpResponse.json(response);
  }),
];
