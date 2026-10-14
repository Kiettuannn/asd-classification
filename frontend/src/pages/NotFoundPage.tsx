import { Link } from 'react-router-dom';

/**
 * Fallback 404 page rendered when no route matches.
 */
function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 text-center">
      <div className="text-8xl font-extrabold text-gray-200">404</div>
      <h1 className="text-2xl font-semibold text-gray-800">Page not found</h1>
      <p className="max-w-sm text-gray-500">
        The page you are looking for doesn't exist or has been moved.
      </p>
      <Link
        to="/"
        className="inline-flex items-center justify-center rounded-md bg-brand-600 px-4 py-2 text-base font-medium text-white transition-colors hover:bg-brand-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2"
      >
        Return home
      </Link>
    </div>
  );
}

export { NotFoundPage };
