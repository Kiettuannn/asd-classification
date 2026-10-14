import { Card, Spinner } from '@/components/ui';
import { useCohortSummary, useModelBenchmarks } from './hooks/useCohortData';
import { BenchmarkTable } from './components/BenchmarkTable';

/**
 * Cohort page — ABIDE demographics summary and model benchmark leaderboard.
 */
function CohortPage() {
  const {
    data: summary,
    isPending: summaryPending,
    error: summaryError,
  } = useCohortSummary();

  const {
    data: benchmarks,
    isPending: benchmarksPending,
    error: benchmarksError,
  } = useModelBenchmarks();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Cohort Analytics</h1>
        <p className="mt-1 text-gray-600">
          ABIDE dataset phenotypic overview and model performance benchmarks.
        </p>
      </div>

      {/* Cohort Summary */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-800">Dataset Summary</h2>
        {summaryPending && <Spinner label="Loading cohort summary…" />}
        {summaryError && (
          <p className="text-red-600 text-sm">Error: {summaryError.message}</p>
        )}
        {summary && (
          <div className="grid gap-4 sm:grid-cols-3">
            <Card>
              <Card.Body>
                <p className="text-3xl font-bold text-brand-700">{summary.total_subjects}</p>
                <p className="mt-1 text-sm text-gray-500">Total Subjects</p>
              </Card.Body>
            </Card>
            <Card>
              <Card.Body>
                <p className="text-3xl font-bold text-red-600">{summary.asd_count}</p>
                <p className="mt-1 text-sm text-gray-500">ASD</p>
              </Card.Body>
            </Card>
            <Card>
              <Card.Body>
                <p className="text-3xl font-bold text-green-600">{summary.td_count}</p>
                <p className="mt-1 text-sm text-gray-500">Typically Developing</p>
              </Card.Body>
            </Card>
          </div>
        )}
      </section>

      {/* Benchmark Leaderboard */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-800">Model Benchmarks</h2>
        {benchmarksPending && <Spinner label="Loading benchmarks…" />}
        {benchmarksError && (
          <p className="text-red-600 text-sm">Error: {benchmarksError.message}</p>
        )}
        {benchmarks && (
          <Card elevated>
            <Card.Body>
              <BenchmarkTable benchmarks={benchmarks} />
            </Card.Body>
          </Card>
        )}
      </section>
    </div>
  );
}

export { CohortPage };
