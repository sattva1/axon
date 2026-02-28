/**
 * Centered empty-state placeholder.
 *
 * Displays an optional icon, a message, and an optional action button.
 * Used when a panel has no data to show.
 */

import type { ReactNode } from 'react';

interface EmptyStateAction {
  label: string;
  onClick: () => void;
}

interface EmptyStateProps {
  /** Optional 24px icon rendered above the message. */
  icon?: ReactNode;
  /** Descriptive text shown in dimmed monospace. */
  message: string;
  /** Optional action button rendered below the message. */
  action?: EmptyStateAction;
}

export function EmptyState({ icon, message, action }: EmptyStateProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        width: '100%',
        height: '100%',
        minHeight: 60,
        padding: 16,
        boxSizing: 'border-box',
      }}
    >
      {icon && (
        <div style={{ color: 'var(--text-dimmed)', fontSize: 24, lineHeight: 1 }}>
          {icon}
        </div>
      )}

      <span
        style={{
          color: 'var(--text-dimmed)',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
          textAlign: 'center',
        }}
      >
        {message}
      </span>

      {action && (
        <button
          type="button"
          onClick={action.onClick}
          style={{
            background: 'var(--accent-dim)',
            color: 'var(--accent)',
            border: 'none',
            borderRadius: 2,
            padding: '4px 12px',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            fontWeight: 500,
            cursor: 'pointer',
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
