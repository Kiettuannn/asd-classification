import { type FormEvent, useState } from 'react';
import { Button, Input, Spinner, Badge, Card } from '@/components/ui';
import { usePredictMutation } from '../hooks/usePredictMutation';

/**
 * Predict form — allows submitting a subject ID for ASD classification.
 * FC matrix entry is deferred to the visualization implementation phase.
 */
function PredictForm() {
  const [subjectId, setSubjectId] = useState('');
  const [validationError, setValidationError] = useState('');
  const { mutate, isPending, data, error, reset } = usePredictMutation();

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setValidationError('');

    if (!subjectId.trim()) {
      setValidationError('Subject ID is required.');
      return;
    }

    reset();
    mutate({ subject_id: subjectId.trim() });
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} noValidate className="space-y-4">
        <Input
          label="Subject ID"
          placeholder="e.g. sub-0050003"
          value={subjectId}
          onChange={(e) => setSubjectId(e.target.value)}
          error={validationError}
          hint="Enter an ABIDE subject identifier."
          disabled={isPending}
        />
        <Button type="submit" isLoading={isPending} disabled={isPending}>
          Run Classification
        </Button>
      </form>

      {/* Loading state */}
      {isPending && (
        <div className="flex justify-center py-8">
          <Spinner size="lg" label="Classifying subject…" />
        </div>
      )}

      {/* Error state */}
      {error && (
        <Card>
          <Card.Body>
            <p className="text-red-600 text-sm">
              Error: {error.message}
            </p>
          </Card.Body>
        </Card>
      )}

      {/* Result state */}
      {data && !isPending && (
        <Card elevated>
          <Card.Header>
            <div className="flex items-center gap-3">
              <span>Classification Result</span>
              <Badge
                label={data.diagnosis}
                variant={data.diagnosis === 'ASD' ? 'error' : 'success'}
              />
            </div>
          </Card.Header>
          <Card.Body>
            <dl className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <dt className="text-gray-500">Subject ID</dt>
                <dd className="font-medium text-gray-900">{data.subject_id ?? '—'}</dd>
              </div>
              <div>
                <dt className="text-gray-500">Confidence</dt>
                <dd className="font-medium text-gray-900">
                  {(data.confidence * 100).toFixed(1)}%
                </dd>
              </div>
              <div>
                <dt className="text-gray-500">ASD Probability</dt>
                <dd className="font-medium text-gray-900">
                  {(data.asd_probability * 100).toFixed(1)}%
                </dd>
              </div>
              <div>
                <dt className="text-gray-500">TD Probability</dt>
                <dd className="font-medium text-gray-900">
                  {(data.td_probability * 100).toFixed(1)}%
                </dd>
              </div>
            </dl>
          </Card.Body>
        </Card>
      )}
    </div>
  );
}

export { PredictForm };
