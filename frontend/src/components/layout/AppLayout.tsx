import { type ReactNode } from 'react';
import { Navbar } from './Navbar';

export interface AppLayoutProps {
  children: ReactNode;
}

/**
 * Root application shell that wraps all page views.
 * Renders the persistent Navbar above a scrollable main content area.
 */
function AppLayout({ children }: AppLayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}

export { AppLayout };
