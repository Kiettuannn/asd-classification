import { type HTMLAttributes } from 'react';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Adds a subtle shadow and border */
  elevated?: boolean;
}

/**
 * Composable card container.
 *
 * @example
 * <Card elevated>
 *   <Card.Header>Title</Card.Header>
 *   <Card.Body>Content</Card.Body>
 * </Card>
 */
function Card({ elevated = false, className = '', children, ...rest }: CardProps) {
  return (
    <div
      className={[
        'rounded-lg bg-white',
        elevated ? 'shadow-md border border-gray-100' : 'border border-gray-200',
        className,
      ].join(' ')}
      {...rest}
    >
      {children}
    </div>
  );
}

function CardHeader({ className = '', children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={['px-6 py-4 border-b border-gray-100 font-semibold text-gray-800', className].join(' ')}
      {...rest}
    >
      {children}
    </div>
  );
}

function CardBody({ className = '', children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={['px-6 py-4', className].join(' ')} {...rest}>
      {children}
    </div>
  );
}

function CardFooter({ className = '', children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={['px-6 py-4 border-t border-gray-100 text-sm text-gray-500', className].join(' ')}
      {...rest}
    >
      {children}
    </div>
  );
}

Card.Header = CardHeader;
Card.Body = CardBody;
Card.Footer = CardFooter;

export { Card };
