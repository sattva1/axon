/**
 * WebGL-rendered graph canvas powered by Sigma.js v3 and Graphology.
 *
 * Initialises the Sigma renderer when the Graphology instance is ready,
 * applies ForceAtlas2 layout via a web worker, and wires up selection,
 * hover, and type-based filtering through the Zustand graph store.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import Sigma from 'sigma';
import FA2LayoutSupervisor from 'graphology-layout-forceatlas2/worker';
import { useGraphStore } from '@/stores/graphStore';
import { useGraph } from '@/hooks/useGraph';
import { cn } from '@/lib/utils';

interface GraphCanvasProps {
  className?: string;
}

/**
 * Core graph visualisation component.
 *
 * Renders the knowledge graph using Sigma.js with WebGL acceleration.
 * The ForceAtlas2 layout runs in a web worker for ~30 seconds, then stops
 * to save CPU. Node/edge visibility, selection, and hover dimming are all
 * handled through nodeReducer/edgeReducer callbacks.
 */
export function GraphCanvas({ className }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const layoutRef = useRef<FA2LayoutSupervisor | null>(null);
  const { graphRef, loading, error } = useGraph();
  const [layoutRunning, setLayoutRunning] = useState(false);

  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const hoveredNodeId = useGraphStore((s) => s.hoveredNodeId);
  const visibleNodeTypes = useGraphStore((s) => s.visibleNodeTypes);
  const visibleEdgeTypes = useGraphStore((s) => s.visibleEdgeTypes);
  const selectNode = useGraphStore((s) => s.selectNode);
  const setHoveredNode = useGraphStore((s) => s.setHoveredNode);

  /** Zoom the camera in by one step. */
  const zoomIn = useCallback(() => {
    const camera = sigmaRef.current?.getCamera();
    if (camera) {
      camera.animatedZoom({ duration: 200 });
    }
  }, []);

  /** Zoom the camera out by one step. */
  const zoomOut = useCallback(() => {
    const camera = sigmaRef.current?.getCamera();
    if (camera) {
      camera.animatedUnzoom({ duration: 200 });
    }
  }, []);

  /** Reset the camera to show the entire graph. */
  const fitToScreen = useCallback(() => {
    const camera = sigmaRef.current?.getCamera();
    if (camera) {
      camera.animatedReset({ duration: 300 });
    }
  }, []);

  /** Toggle the ForceAtlas2 layout on/off. */
  const toggleLayout = useCallback(() => {
    const layout = layoutRef.current;
    if (!layout) return;

    if (layout.isRunning()) {
      layout.stop();
      setLayoutRunning(false);
    } else {
      layout.start();
      setLayoutRunning(true);
    }
  }, []);

  // Initialise Sigma and ForceAtlas2 when the Graphology graph is ready.
  useEffect(() => {
    const container = containerRef.current;
    const graph = graphRef.current;
    if (!container || !graph) return;

    // Snapshot the current store values for the reducers. The refresh effect
    // below triggers sigma.refresh() whenever these change, which causes
    // Sigma to re-invoke the reducers with fresh closure values.
    const sigma = new Sigma(graph, container, {
      renderLabels: true,
      labelFont: 'JetBrains Mono, monospace',
      labelSize: 11,
      labelColor: { color: '#c5ced6' },
      defaultEdgeColor: '#3d4f5f',
      defaultNodeColor: '#6b7d8e',
      labelRenderedSizeThreshold: 6,

      nodeReducer: (node, data) => {
        const res = { ...data };
        const nodeType = (data.nodeType ?? '') as string;

        // Retrieve current store state directly for the reducer.
        const state = useGraphStore.getState();

        // Hide node types that are filtered out.
        if (!state.visibleNodeTypes.has(nodeType)) {
          res.hidden = true;
          return res;
        }

        // Dim unrelated nodes when a node is selected.
        if (state.selectedNodeId && node !== state.selectedNodeId) {
          const isNeighbor =
            graph.hasEdge(state.selectedNodeId, node) ||
            graph.hasEdge(node, state.selectedNodeId);
          if (!isNeighbor) {
            res.color = '#1a2030';
            res.label = '';
          }
        }

        // Highlight on hover.
        if (state.hoveredNodeId && node === state.hoveredNodeId) {
          res.highlighted = true;
        }

        return res;
      },

      edgeReducer: (edge, data) => {
        const res = { ...data };
        const edgeType = (data.edgeType ?? '') as string;

        const state = useGraphStore.getState();

        // Hide edge types that are filtered out.
        if (!state.visibleEdgeTypes.has(edgeType)) {
          res.hidden = true;
          return res;
        }

        // When a node is selected, brighten connected edges, dim the rest.
        if (state.selectedNodeId) {
          const source = graph.source(edge);
          const target = graph.target(edge);
          if (source !== state.selectedNodeId && target !== state.selectedNodeId) {
            res.color = '#0a0e14';
          } else {
            res.size = 2;
          }
        }

        return res;
      },
    });

    sigmaRef.current = sigma;

    // Wire up interaction events.
    sigma.on('clickNode', ({ node }) => {
      selectNode(node);
    });

    sigma.on('clickStage', () => {
      selectNode(null);
    });

    sigma.on('enterNode', ({ node }) => {
      setHoveredNode(node);
    });

    sigma.on('leaveNode', () => {
      setHoveredNode(null);
    });

    // Start ForceAtlas2 layout in a web worker.
    const layout = new FA2LayoutSupervisor(graph, {
      settings: {
        gravity: 1,
        scalingRatio: 2,
        barnesHutOptimize: true,
        slowDown: 5,
      },
    });
    layout.start();
    layoutRef.current = layout;
    setLayoutRunning(true);

    // Stop the layout after 30 seconds to save CPU.
    const timer = setTimeout(() => {
      if (layout.isRunning()) {
        layout.stop();
        setLayoutRunning(false);
      }
    }, 30_000);

    return () => {
      clearTimeout(timer);
      layout.kill();
      sigma.kill();
      sigmaRef.current = null;
      layoutRef.current = null;
      setLayoutRunning(false);
    };
  }, [loading, selectNode, setHoveredNode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-render Sigma when filters, selection, or hover state change.
  useEffect(() => {
    sigmaRef.current?.refresh();
  }, [selectedNodeId, hoveredNodeId, visibleNodeTypes, visibleEdgeTypes]);

  if (error) {
    return (
      <div
        className={cn('flex items-center justify-center h-full', className)}
        style={{ color: 'var(--danger)' }}
      >
        Failed to load graph: {error}
      </div>
    );
  }

  if (loading) {
    return (
      <div
        className={cn('flex items-center justify-center h-full', className)}
        style={{ color: 'var(--text-secondary)' }}
      >
        <span style={{ color: 'var(--accent)' }}>&#x25CF;</span>
        &nbsp;Loading graph...
      </div>
    );
  }

  return (
    <div className={cn('relative w-full h-full', className)} style={{ background: 'var(--bg-primary)' }}>
      <div ref={containerRef} className="w-full h-full" />
      <GraphControls
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onFitToScreen={fitToScreen}
        onToggleLayout={toggleLayout}
        layoutRunning={layoutRunning}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline GraphControls (small enough to co-locate)
// ---------------------------------------------------------------------------

interface GraphControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToScreen: () => void;
  onToggleLayout: () => void;
  layoutRunning: boolean;
}

/**
 * Floating control bar at the bottom-left of the graph canvas.
 *
 * Four small buttons stacked vertically: zoom in, zoom out, fit to screen,
 * and play/pause the force-directed layout.
 */
function GraphControls({
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
        <PlusIcon />
      </ControlButton>
      <ControlButton onClick={onZoomOut} title="Zoom out" aria-label="Zoom out">
        <MinusIcon />
      </ControlButton>
      <ControlButton onClick={onFitToScreen} title="Fit to screen" aria-label="Fit to screen">
        <MaximizeIcon />
      </ControlButton>
      <ControlButton onClick={onToggleLayout} title={layoutRunning ? 'Pause layout' : 'Resume layout'} aria-label={layoutRunning ? 'Pause layout' : 'Resume layout'}>
        {layoutRunning ? <PauseIcon /> : <PlayIcon />}
      </ControlButton>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Control button wrapper
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

// ---------------------------------------------------------------------------
// Inline SVG icons (12x12) to avoid importing lucide-react for 4 tiny icons
// ---------------------------------------------------------------------------

function PlusIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
      <line x1="6" y1="2" x2="6" y2="10" />
      <line x1="2" y1="6" x2="10" y2="6" />
    </svg>
  );
}

function MinusIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
      <line x1="2" y1="6" x2="10" y2="6" />
    </svg>
  );
}

function MaximizeIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="2" y="2" width="8" height="8" rx="0.5" />
      <line x1="4" y1="4" x2="4" y2="4.01" strokeLinecap="round" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" stroke="none">
      <polygon points="3,1.5 10,6 3,10.5" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" stroke="none">
      <rect x="2.5" y="2" width="2.5" height="8" rx="0.5" />
      <rect x="7" y="2" width="2.5" height="8" rx="0.5" />
    </svg>
  );
}
