import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { AppLayout } from '@/components/layout';
import { Spinner } from '@/components/ui';
import { HomePage } from '@/pages/HomePage';
import { NotFoundPage } from '@/pages/NotFoundPage';

// Code-split each feature page to keep the initial bundle small
const PredictPage    = lazy(() => import('@/features/predict/PredictPage').then((m) => ({ default: m.PredictPage })));
const ConnectomePage = lazy(() => import('@/features/connectome/ConnectomePage').then((m) => ({ default: m.ConnectomePage })));
const CohortPage     = lazy(() => import('@/features/cohort/CohortPage').then((m) => ({ default: m.CohortPage })));

/** Full-page loading fallback while lazy chunks are downloading */
function PageFallback() {
  return (
    <div className="flex h-64 items-center justify-center">
      <Spinner size="lg" label="Loading page…" />
    </div>
  );
}

/**
 * Application route tree.
 *
 * Routes:
 *   /              → HomePage
 *   /predict       → PredictPage
 *   /connectome    → ConnectomePage
 *   /cohort        → CohortPage
 *   *              → NotFoundPage
 */
function AppRoutes() {
  return (
    <AppLayout>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/"           element={<HomePage />} />
          <Route path="/predict"    element={<PredictPage />} />
          <Route path="/connectome" element={<ConnectomePage />} />
          <Route path="/cohort"     element={<CohortPage />} />
          <Route path="*"           element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </AppLayout>
  );
}

export { AppRoutes };
