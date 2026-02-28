/**
 * Converts raw API graph data into a Graphology instance for Sigma.js rendering.
 *
 * Assigns visual attributes (color, size, position) to each node and edge so
 * that Sigma can render the graph without additional processing.
 */

import { MultiDirectedGraph } from 'graphology';
import type { GraphNode, GraphEdge, NodeLabel } from '@/types';

/** Color palette for each node label. Maps to CSS variable equivalents. */
const NODE_COLORS: Record<string, string> = {
  function: '#39d353',
  class: '#58a6ff',
  method: '#a371f7',
  interface: '#3fb8af',
  type_alias: '#56d4dd',
  enum: '#f0883e',
  file: '#6b7d8e',
  folder: '#4d5969',
  community: '#a371f7',
  process: '#3fb8af',
};

const DEFAULT_NODE_COLOR = '#6b7d8e';
const DEFAULT_EDGE_COLOR = '#3d4f5f';

/**
 * Build a Graphology MultiDirectedGraph from raw API node/edge arrays.
 *
 * Nodes receive random initial positions (ForceAtlas2 will reposition them),
 * colors based on their label, and sizes based on their degree.
 *
 * Edges that reference missing nodes or duplicate keys are silently skipped
 * to tolerate inconsistent backend data.
 *
 * @param nodes - Array of graph nodes from the API.
 * @param edges - Array of graph edges from the API.
 * @returns A fully-attributed Graphology graph ready for Sigma.
 */
export function buildGraphology(nodes: GraphNode[], edges: GraphEdge[]): MultiDirectedGraph {
  const graph = new MultiDirectedGraph();

  for (const node of nodes) {
    graph.addNode(node.id, {
      label: node.name,
      x: Math.random() * 1000,
      y: Math.random() * 1000,
      size: 4,
      color: NODE_COLORS[node.label] ?? DEFAULT_NODE_COLOR,
      nodeType: node.label as NodeLabel,
      filePath: node.filePath,
      startLine: node.startLine,
      endLine: node.endLine,
      signature: node.signature,
      language: node.language,
      className: node.className,
      isDead: node.isDead,
      isEntryPoint: node.isEntryPoint,
      isExported: node.isExported,
    });
  }

  for (const edge of edges) {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) {
      continue;
    }
    try {
      graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
        edgeType: edge.type,
        color: DEFAULT_EDGE_COLOR,
        size: 1,
        confidence: edge.confidence,
        strength: edge.strength,
        stepNumber: edge.stepNumber,
      });
    } catch {
      // Skip duplicate edge keys silently.
    }
  }

  // Assign node sizes proportional to degree after all edges are added.
  graph.forEachNode((id) => {
    const degree = graph.degree(id);
    graph.setNodeAttribute(id, 'size', 4 + Math.min(16, Math.sqrt(degree) * 2));
  });

  return graph;
}
