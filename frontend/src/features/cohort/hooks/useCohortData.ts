/**
 * TanStack Query hooks for the cohort feature.
 */
import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import type { CohortSummary, ModelBenchmark } from '@/types/cohort';
import { fetchCohortSummary, fetchModelBenchmarks } from '../api/cohortApi';

export const COHORT_QUERY_KEYS = {
  summary: ['cohort', 'summary'] as const,
  benchmarks: ['cohort', 'benchmarks'] as const,
};

/**
 * Query hook for fetching the ABIDE cohort summary statistics.
 */
export function useCohortSummary(): UseQueryResult<CohortSummary, Error> {
  return useQuery<CohortSummary, Error>({
    queryKey: COHORT_QUERY_KEYS.summary,
    queryFn: fetchCohortSummary,
  });
}

/**
 * Query hook for fetching the model benchmark leaderboard.
 */
export function useModelBenchmarks(): UseQueryResult<ModelBenchmark[], Error> {
  return useQuery<ModelBenchmark[], Error>({
    queryKey: COHORT_QUERY_KEYS.benchmarks,
    queryFn: fetchModelBenchmarks,
  });
}
