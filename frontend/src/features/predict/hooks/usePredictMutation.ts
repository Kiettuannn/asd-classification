/**
 * TanStack Query hooks for the predict feature.
 */
import { useMutation, type UseMutationResult } from '@tanstack/react-query';
import type { PredictRequest, PredictResponse } from '@/types/api';
import { postPredict } from '../api/predictApi';

/**
 * Mutation hook for submitting a classification request.
 *
 * @example
 * const { mutate, isPending, data, error } = usePredictMutation();
 * mutate({ subject_id: 'sub-0050003' });
 */
export function usePredictMutation(): UseMutationResult<PredictResponse, Error, PredictRequest> {
  return useMutation<PredictResponse, Error, PredictRequest>({
    mutationFn: postPredict,
  });
}
