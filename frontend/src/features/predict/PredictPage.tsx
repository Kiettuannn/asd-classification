/**
 * Predict feature page — wraps the PredictForm.
 */
import { PredictForm } from './components/PredictForm';

function PredictPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">ASD Classification</h1>
        <p className="mt-1 text-gray-600">
          Submit a subject ID or functional connectivity matrix to classify ASD vs TD.
        </p>
      </div>
      <PredictForm />
    </div>
  );
}

export { PredictPage };
