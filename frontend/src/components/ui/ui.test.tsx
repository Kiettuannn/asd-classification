/**
 * Tests: Shared UI primitives — Button, Badge, Input, Spinner, Card
 *
 * Checks rendering, accessible attributes, and variant styling.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from './Button';
import { Badge } from './Badge';
import { Input } from './Input';
import { Spinner } from './Spinner';
import { Card } from './Card';

// ─── Button ────────────────────────────────────────────────────────────────
describe('Button', () => {
  it('renders its label text', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button', { name: /Click me/i })).toBeInTheDocument();
  });

  it('is disabled and aria-disabled when isLoading is true', () => {
    render(<Button isLoading>Submitting</Button>);
    const btn = screen.getByRole('button');
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('aria-disabled', 'true');
  });

  it('fires onClick when not disabled', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Go</Button>);
    await user.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('does not fire onClick when disabled', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<Button disabled onClick={onClick}>Go</Button>);
    await user.click(screen.getByRole('button'));
    expect(onClick).not.toHaveBeenCalled();
  });
});

// ─── Badge ─────────────────────────────────────────────────────────────────
describe('Badge', () => {
  it('renders the label', () => {
    render(<Badge label="ASD" />);
    expect(screen.getByText('ASD')).toBeInTheDocument();
  });

  it('applies error variant classes for ASD diagnosis', () => {
    render(<Badge label="ASD" variant="error" />);
    const badge = screen.getByText('ASD');
    expect(badge.className).toContain('bg-red-100');
  });

  it('applies success variant classes for TD diagnosis', () => {
    render(<Badge label="TD" variant="success" />);
    const badge = screen.getByText('TD');
    expect(badge.className).toContain('bg-green-100');
  });
});

// ─── Input ─────────────────────────────────────────────────────────────────
describe('Input', () => {
  it('renders with a label associated via htmlFor', () => {
    render(<Input label="Subject ID" />);
    expect(screen.getByLabelText(/Subject ID/i)).toBeInTheDocument();
  });

  it('shows an error message with role="alert"', () => {
    render(<Input label="Subject ID" error="Required" />);
    expect(screen.getByRole('alert')).toHaveTextContent('Required');
  });

  it('sets aria-invalid when there is an error', () => {
    render(<Input label="Subject ID" error="Required" />);
    expect(screen.getByLabelText(/Subject ID/i)).toHaveAttribute('aria-invalid', 'true');
  });
});

// ─── Spinner ───────────────────────────────────────────────────────────────
describe('Spinner', () => {
  it('has a status role for screen readers', () => {
    render(<Spinner label="Loading…" />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('renders the sr-only label text', () => {
    render(<Spinner label="Classifying subject" />);
    expect(screen.getByText('Classifying subject')).toBeInTheDocument();
  });
});

// ─── Card ──────────────────────────────────────────────────────────────────
describe('Card', () => {
  it('renders children inside Card.Body', () => {
    render(
      <Card>
        <Card.Body>Hello card</Card.Body>
      </Card>,
    );
    expect(screen.getByText('Hello card')).toBeInTheDocument();
  });

  it('renders header and footer text', () => {
    render(
      <Card>
        <Card.Header>My Header</Card.Header>
        <Card.Body>Content</Card.Body>
        <Card.Footer>Footer note</Card.Footer>
      </Card>,
    );
    expect(screen.getByText('My Header')).toBeInTheDocument();
    expect(screen.getByText('Footer note')).toBeInTheDocument();
  });
});
