import type { ConnectionWeight } from '@/types/api';

interface ConnectionTableProps {
  connections: ConnectionWeight[];
}

/**
 * Stub: Renders a tabular list of the top salient brain connections.
 * Visualization (heatmap, graph) is deferred to the implementation phase.
 */
function ConnectionTable({ connections }: ConnectionTableProps) {
  if (connections.length === 0) {
    return <p className="text-gray-500 text-sm">No connections to display.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-gray-500">
            <th className="pb-2 pr-4 font-medium">Rank</th>
            <th className="pb-2 pr-4 font-medium">ROI Source</th>
            <th className="pb-2 pr-4 font-medium">ROI Target</th>
            <th className="pb-2 font-medium">Weight</th>
          </tr>
        </thead>
        <tbody>
          {connections.map((conn, idx) => (
            <tr
              key={`${conn.roi_source}-${conn.roi_target}`}
              className="border-b border-gray-100 hover:bg-gray-50"
            >
              <td className="py-2 pr-4 text-gray-500">{idx + 1}</td>
              <td className="py-2 pr-4 font-mono text-gray-800">ROI {conn.roi_source}</td>
              <td className="py-2 pr-4 font-mono text-gray-800">ROI {conn.roi_target}</td>
              <td className="py-2 text-gray-800">{conn.weight.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export { ConnectionTable };
