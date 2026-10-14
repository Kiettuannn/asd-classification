import { type FormEvent, useState } from 'react';
import { Button, Card, Input, Spinner } from '@/components/ui';
import { useConnectomeExplainMutation } from './hooks/useConnectomeExplainMutation';
import { ConnectionTable } from './components/ConnectionTable';

/**
 * Connectome page — explore top salient brain connections for a subject.
 */
function ConnectomePage() {
  const [subjectId, setSubjectId] = useState('');
  const [validationError, setValidationError] = useState('');
  const { mutate, isPending, data, error, reset } = useConnectomeExplainMutation();

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
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Connectome Explorer</h1>
        <p className="mt-1 text-gray-600">
          Visualize top salient functional connections driving the classification decision.
        </p>
      </div>

      <form onSubmit={handleSubmit} noValidate className="flex items-end gap-4">
        <Input
          label="Subject ID"
          placeholder="e.g. sub-0050003"
          value={subjectId}
          onChange={(e) => setSubjectId(e.target.value)}
          error={validationError}
          disabled={isPending}
        />
        <Button type="submit" isLoading={isPending} disabled={isPending}>
          Load Connectome
        </Button>
      </form>

      {isPending && (
        <div className="flex justify-center py-8">
          <Spinner size="lg" label="Loading connectome data…" />
        </div>
      )}

      {error && (
        <Card>
          <Card.Body>
            <p className="text-red-600 text-sm">Error: {error.message}</p>
          </Card.Body>
        </Card>
      )}

      {data && !isPending && (
        <Card elevated>
          <Card.Header>
            Top Connections — {data.model_name}
          </Card.Header>
          <Card.Body>
            <ConnectionTable connections={data.top_connections} />
          </Card.Body>
          <Card.Footer>
            {data.top_connections.length} connections · Subject: {data.subject_id ?? '—'}
          </Card.Footer>
        </Card>
      )}
    </div>
  );
}

export { ConnectomePage };
