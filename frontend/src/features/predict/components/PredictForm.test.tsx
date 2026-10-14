/**
 * Tests: PredictForm
 *
 * Verifies form validation and that a valid submission fires the /predict mutation.
 * Uses MSW to intercept the HTTP call at the network boundary.
 */
import { describe, it, expect } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '@/test/utils';
import { PredictForm } from '@/features/predict/components/PredictForm';

describe('PredictForm', () => {
  it('renders the Subject ID input and submit button', () => {
    renderWithProviders(<PredictForm />);
    expect(screen.getByLabelText(/Subject ID/i)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Run Classification/i }),
    ).toBeInTheDocument();
  });

  it('shows a validation error when submitting with an empty subject ID', async () => {
    const user = userEvent.setup();
    renderWithProviders(<PredictForm />);

    await user.click(screen.getByRole('button', { name: /Run Classification/i }));

    expect(
      await screen.findByRole('alert'),
    ).toHaveTextContent(/Subject ID is required/i);
  });

  it('calls the predict API and renders the result card on valid submission', async () => {
    const user = userEvent.setup();
    renderWithProviders(<PredictForm />);

    await user.type(screen.getByLabelText(/Subject ID/i), 'sub-0050003');
    await user.click(screen.getByRole('button', { name: /Run Classification/i }));

    // MSW returns diagnosis: 'ASD', asd_probability: 0.82
    await waitFor(() => {
      expect(screen.getByText(/Classification Result/i)).toBeInTheDocument();
    });

    expect(screen.getByText('ASD')).toBeInTheDocument();
    // Both "Confidence" and "ASD Probability" render 82.0% in the fixture (both = 0.82)
    const percentCells = screen.getAllByText(/82\.0%/);
    expect(percentCells.length).toBeGreaterThanOrEqual(1);
  });
});
