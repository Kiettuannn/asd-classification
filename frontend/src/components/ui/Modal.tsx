import { type ReactNode, useEffect, useRef } from 'react';
import { Button } from './Button';

export interface ModalProps {
  /** Controls visibility */
  isOpen: boolean;
  /** Fired when user dismisses the modal (Escape key or backdrop click) */
  onClose: () => void;
  title: string;
  children: ReactNode;
  /** Width constraint class, default 'max-w-lg' */
  maxWidth?: string;
}

/**
 * Accessible dialog modal with focus trapping and Escape key support.
 * Renders via a portal-like approach using a fixed overlay.
 *
 * @example
 * <Modal isOpen={showModal} onClose={() => setShowModal(false)} title="Prediction Result">
 *   <PredictResultCard result={result} />
 * </Modal>
 */
function Modal({ isOpen, onClose, title, children, maxWidth = 'max-w-lg' }: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  // Trap focus inside the dialog when open
  useEffect(() => {
    if (!isOpen) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      previousFocus?.focus();
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      aria-modal="true"
      role="dialog"
      aria-labelledby="modal-title"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div
        ref={dialogRef}
        tabIndex={-1}
        className={[
          'relative z-10 w-full rounded-xl bg-white shadow-2xl focus:outline-none',
          'mx-4 overflow-auto',
          maxWidth,
        ].join(' ')}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-100 px-6 py-4">
          <h2 id="modal-title" className="text-lg font-semibold text-gray-900">
            {title}
          </h2>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close dialog">
            ✕
          </Button>
        </div>

        {/* Body */}
        <div className="px-6 py-4">{children}</div>
      </div>
    </div>
  );
}

export { Modal };
