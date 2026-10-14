/**
 * Cohort feature API module.
 * Placeholder functions for fetching cohort phenotypic data and benchmarks.
 * Real endpoints will be added during the cohort analytics implementation phase.
 */
import apiClient from '@/api/client';
import type { CohortSummary, ModelBenchmark } from '@/types/cohort';

/**
 * GET /cohort/summary – Aggregate statistics for the loaded ABIDE cohort.
 * @stub – endpoint to be implemented in the backend.
 */
export async function fetchCohortSummary(): Promise<CohortSummary> {
  const { data } = await apiClient.get<CohortSummary>('/cohort/summary');
  return data;
}

/**
 * GET /cohort/benchmarks – Model performance leaderboard.
 * @stub – endpoint to be implemented in the backend.
 */
export async function fetchModelBenchmarks(): Promise<ModelBenchmark[]> {
  const { data } = await apiClient.get<ModelBenchmark[]>('/cohort/benchmarks');
  return data;
}
