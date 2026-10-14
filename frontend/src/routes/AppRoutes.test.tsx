/**
 * Tests: AppRoutes navigation
 *
 * Verifies that navigating to each route renders the correct page heading.
 * Tests run against the full route tree wrapped in all providers.
 */
import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '@/test/utils';
import { AppRoutes } from '@/routes/AppRoutes';

describe('AppRoutes', () => {
  it('renders the Home page at /', async () => {
    renderWithProviders(<AppRoutes />);
    expect(
      await screen.findByText(/ASD Classification & Connectome Dashboard/i),
    ).toBeInTheDocument();
  });

  it('navigates to /predict and renders the Predict page heading', async () => {
    const user = userEvent.setup();
    renderWithProviders(<AppRoutes />);

    await user.click(await screen.findByRole('link', { name: /predict/i }));

    expect(
      await screen.findByRole('heading', { name: /ASD Classification/i }),
    ).toBeInTheDocument();
  });

  it('navigates to /connectome and renders the Connectome page heading', async () => {
    const user = userEvent.setup();
    renderWithProviders(<AppRoutes />);

    await user.click(await screen.findByRole('link', { name: /connectome/i }));

    expect(
      await screen.findByRole('heading', { name: /Connectome Explorer/i }),
    ).toBeInTheDocument();
  });

  it('navigates to /cohort and renders the Cohort page heading', async () => {
    const user = userEvent.setup();
    renderWithProviders(<AppRoutes />);

    await user.click(await screen.findByRole('link', { name: /cohort/i }));

    expect(
      await screen.findByRole('heading', { name: /Cohort Analytics/i }),
    ).toBeInTheDocument();
  });

  it('renders a 404 page for an unknown route', async () => {
    // Render at an unknown path by overriding initialEntries via MemoryRouter
    // We use window.history here since BrowserRouter reads window.location
    window.history.pushState({}, '', '/this-route-does-not-exist');
    renderWithProviders(<AppRoutes />);

    expect(await screen.findByText(/Page not found/i)).toBeInTheDocument();
  });
});
