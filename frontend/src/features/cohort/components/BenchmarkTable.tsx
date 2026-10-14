import type { ModelBenchmark } from '@/types/cohort';

interface BenchmarkTableProps {
  benchmarks: ModelBenchmark[];
}

/**
 * Stub: Renders a leaderboard table of model performance metrics.
 */
function BenchmarkTable({ benchmarks }: BenchmarkTableProps) {
  if (benchmarks.length === 0) {
    return <p className="text-gray-500 text-sm">No benchmark data available.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-gray-500">
            <th className="pb-2 pr-4 font-medium">Model</th>
            <th className="pb-2 pr-4 font-medium">Accuracy</th>
            <th className="pb-2 pr-4 font-medium">Sensitivity</th>
            <th className="pb-2 pr-4 font-medium">Specificity</th>
            <th className="pb-2 font-medium">AUC-ROC</th>
          </tr>
        </thead>
        <tbody>
          {benchmarks.map((bm) => (
            <tr
              key={bm.model_name}
              className="border-b border-gray-100 hover:bg-gray-50"
            >
              <td className="py-2 pr-4 font-medium text-gray-800">{bm.model_name}</td>
              <td className="py-2 pr-4 text-gray-700">{(bm.accuracy * 100).toFixed(1)}%</td>
              <td className="py-2 pr-4 text-gray-700">{(bm.sensitivity * 100).toFixed(1)}%</td>
              <td className="py-2 pr-4 text-gray-700">{(bm.specificity * 100).toFixed(1)}%</td>
              <td className="py-2 text-gray-700">{bm.auc_roc.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export { BenchmarkTable };
