import { Card } from '@/components/ui';

/**
 * Landing / home page — overview of the dashboard capabilities.
 */
function HomePage() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold text-gray-900">
          ASD Classification &amp; Connectome Dashboard
        </h1>
        <p className="mt-2 text-gray-600 max-w-2xl">
          Functional connectivity–based Autism Spectrum Disorder classification using
          Graph Neural Networks trained on the ABIDE dataset.
        </p>
      </section>

      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <Card elevated>
          <Card.Header>🔬 Predict</Card.Header>
          <Card.Body>
            Submit a subject ID or FC matrix to receive an ASD vs TD classification
            with per-class probabilities.
          </Card.Body>
        </Card>

        <Card elevated>
          <Card.Header>🕸️ Connectome</Card.Header>
          <Card.Body>
            Explore brain network graphs, FC correlation heatmaps, and the top salient
            inter-ROI connections driving classification decisions.
          </Card.Body>
        </Card>

        <Card elevated>
          <Card.Header>📊 Cohort</Card.Header>
          <Card.Body>
            Browse ABIDE phenotypic demographics, site breakdowns, and model benchmark
            performance comparisons.
          </Card.Body>
        </Card>
      </div>
    </div>
  );
}

export { HomePage };
