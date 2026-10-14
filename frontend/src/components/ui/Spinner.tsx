export type SpinnerSize = 'sm' | 'md' | 'lg';

export interface SpinnerProps {
  size?: SpinnerSize;
  label?: string;
  className?: string;
}

const sizeClasses: Record<SpinnerSize, string> = {
  sm: 'h-4 w-4 border-2',
  md: 'h-8 w-8 border-[3px]',
  lg: 'h-12 w-12 border-4',
};

/**
 * Animated loading spinner with an accessible screen-reader label.
 *
 * @example
 * <Spinner size="md" label="Classifying subject..." />
 */
function Spinner({ size = 'md', label = 'Loading…', className = '' }: SpinnerProps) {
  return (
    <div role="status" className={['inline-flex flex-col items-center gap-2', className].join(' ')}>
      <div
        className={[
          'animate-spin rounded-full border-gray-300 border-t-brand-600',
          sizeClasses[size],
        ].join(' ')}
        aria-hidden="true"
      />
      <span className="sr-only">{label}</span>
    </div>
  );
}

export { Spinner };
