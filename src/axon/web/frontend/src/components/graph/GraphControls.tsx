/**
 * Floating graph control buttons for the canvas overlay.
 *
 * Four small (24x24px) buttons stacked vertically at the bottom-left:
 *   + zoom in
 *   - zoom out
 *   [ ] fit to screen
 *   |>/|| play/pause layout
 *
 * Exported separately so it can be used standalone or composed into
 * other layouts. The GraphCanvas component also includes an inline
 * version of these controls.
 */

import { ZoomIn, ZoomOut, Maximize, Play, Pause } from 'lucide-react';

interface GraphControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToScreen: () => void;
  onToggleLayout: () => void;
  layoutRunning: boolean;
}

export function GraphControls({
  onZoomIn,
  onZoomOut,
  onFitToScreen,
  onToggleLayout,
  layoutRunning,
}: GraphControlsProps) {
  return (
    <div
      className="absolute bottom-3 left-3 flex flex-col gap-1"
      style={{ zIndex: 10 }}
    >
      <ControlButton onClick={onZoomIn} title="Zoom in" aria-label="Zoom in">
        <ZoomIn size={12} />
      </ControlButton>
      <ControlButton onClick={onZoomOut} title="Zoom out" aria-label="Zoom out">
        <ZoomOut size={12} />
      </ControlButton>
      <ControlButton onClick={onFitToScreen} title="Fit to screen" aria-label="Fit to screen">
        <Maximize size={12} />
      </ControlButton>
      <ControlButton
        onClick={onToggleLayout}
        title={layoutRunning ? 'Pause layout' : 'Resume layout'}
        aria-label={layoutRunning ? 'Pause layout' : 'Resume layout'}
      >
        {layoutRunning ? <Pause size={12} /> : <Play size={12} />}
      </ControlButton>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Internal control button wrapper
// ---------------------------------------------------------------------------

interface ControlButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
}

function ControlButton({ children, ...props }: ControlButtonProps) {
  return (
    <button
      type="button"
      className="flex items-center justify-center transition-colors"
      style={{
        width: 24,
        height: 24,
        background: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderRadius: 2,
        color: 'var(--text-secondary)',
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.color = 'var(--accent)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-secondary)';
      }}
      {...props}
    >
      {children}
    </button>
  );
}
