/**
 * TanStack Query hooks for the connectome feature.
 */
import { useMutation, type UseMutationResult } from '@tanstack/react-query';
import type { PredictRequest, ExplainResponse } from '@/types/api';
import { fetchConnectomeExplain } from '../api/connectomeApi';

/**
 * Mutation hook for fetching connectome explainability data.
 *
 * @example
 * const { mutate, isPending, data } = useConnectomeExplainMutation();
 * mutate({ subject_id: 'sub-0050003' });
 */
export function useConnectomeExplainMutation(): UseMutationResult<
  ExplainResponse,
  Error,
  PredictRequest
> {
  return useMutation<ExplainResponse, Error, PredictRequest>({
    mutationFn: fetchConnectomeExplain,
  });
}
