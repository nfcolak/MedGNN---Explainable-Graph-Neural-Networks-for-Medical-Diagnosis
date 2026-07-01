import cytoscape, { Core, ElementDefinition, NodeSingular, StylesheetCSS } from 'cytoscape';
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { PatientGraph, PatientGraphNode, TooltipState } from '../types/graph';

export interface GraphCanvasHandle {
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
  fit: () => void;
}

interface GraphCanvasProps {
  graph: PatientGraph;
  searchTerm: string;
  onNodeSelect: (node: PatientGraphNode | null) => void;
}

const TYPE_COLORS = {
  patient: '#ef4444',
  vital: '#38bdf8',
  medication: '#4ade80',
  chief_complaint: '#fb923c',
  diagnosis_attribute: '#c084fc',
  symptom: '#a78bfa',
  icd: '#f59e0b',
};

const TYPE_LABELS: Record<string, string> = {
  patient: 'Patient',
  vital: 'Vital sign',
  medication: 'Medication',
  chief_complaint: 'Chief complaint',
  diagnosis_attribute: 'Diagnosis attribute',
  symptom: 'Symptom',
  icd: 'Diagnosis code',
};

function normalizeType(type: string) {
  if (type === 'med') return 'medication';
  if (type === 'cc') return 'chief_complaint';
  if (type === 'PATIENT') return 'patient';
  return type;
}

function formatValue(value: unknown) {
  if (typeof value === 'number') return `${value >= 0 ? '+' : ''}${value.toFixed(1)}`;
  if (typeof value === 'string' && value.length > 0) return value;
  return '';
}

function nodeValue(node: PatientGraphNode) {
  return node.value ?? node.z;
}

function wrapLabel(label: string) {
  if (label.includes(' ')) return label.replace(/\s+/g, '\n');
  if (label.length <= 11) return label;
  const midpoint = Math.ceil(label.length / 2);
  const splitAt = Math.min(label.length - 4, Math.max(7, midpoint));
  return `${label.slice(0, splitAt)}\n${label.slice(splitAt)}`;
}

function displayLabel(node: PatientGraphNode) {
  const value = formatValue(nodeValue(node));
  const label = wrapLabel(node.label);
  return value ? `${label}\n${value}` : label;
}

function edgeEndpoints(edge: PatientGraph['edges'][number]) {
  return {
    source: String(edge.source ?? edge.a),
    target: String(edge.target ?? edge.b),
  };
}

function edgeType(edge: PatientGraph['edges'][number]) {
  const type = edge.type ?? edge.kind;
  if (type === 'pmi') return 'pmi_bundle';
  if (type === 'star') return 'patient_feature';
  return type ?? 'patient_feature';
}

function ringForNode(node: PatientGraphNode) {
  const type = normalizeType(node.type);
  if (type === 'patient') return 0;
  if (type === 'vital') return 0.22;
  if (type === 'chief_complaint') return 0.36;
  if (type === 'diagnosis_attribute' || type === 'symptom' || type === 'icd') return 0.43;
  return 0.52;
}

function positionsForGraph(graph: PatientGraph, width: number, height: number) {
  const center = { x: width / 2, y: height / 2 };
  const base = Math.max(190, Math.min(width, height) * 0.72);
  const groups = new Map<number, PatientGraphNode[]>();

  graph.nodes.forEach((node) => {
    const ring = ringForNode(node);
    groups.set(ring, [...(groups.get(ring) ?? []), node]);
  });

  const offsets = new Map<number, number>([
    [0.22, -116],
    [0.36, 18],
    [0.43, -18],
    [0.52, 98],
  ]);
  const positions = new Map<string, { x: number; y: number }>();

  graph.nodes.forEach((node) => {
    if (ringForNode(node) === 0) positions.set(String(node.id), center);
  });

  [...groups.entries()]
    .filter(([ring]) => ring > 0)
    .forEach(([ring, nodes]) => {
      const radius = base * ring;
      const start = offsets.get(ring) ?? 0;
      nodes.forEach((node, index) => {
        const angle = ((start + (360 / nodes.length) * index) * Math.PI) / 180;
        const type = normalizeType(node.type);
        const yBias = type === 'medication' ? 18 : type === 'vital' ? -6 : 4;
        positions.set(String(node.id), {
          x: center.x + Math.cos(angle) * radius,
          y: center.y + Math.sin(angle) * radius + yBias,
        });
      });
    });

  return positions;
}

const cytoscapeStyle = [
  {
    selector: 'core',
    css: {
      'active-bg-color': '#38bdf8',
      'active-bg-opacity': 0.12,
      'selection-box-color': '#38bdf8',
      'selection-box-opacity': 0.1,
    },
  },
  {
    selector: 'node',
    css: {
      width: 'mapData(size, 0, 1, 42, 78)',
      height: 'mapData(size, 0, 1, 42, 78)',
      'background-color': 'data(color)',
      'border-color': '#f8fafc',
      'border-width': 1.3,
      'border-opacity': 0.82,
      color: '#f8fafc',
      label: 'data(displayLabel)',
      'font-family': 'Inter, ui-sans-serif, system-ui',
      'font-size': 'data(fontSize)',
      'font-weight': 700,
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': 86,
      'text-overflow-wrap': 'anywhere',
      'line-height': 1.08,
      'text-outline-color': '#020617',
      'text-outline-width': 3,
      'overlay-opacity': 0,
      'transition-property': 'border-width, border-color, opacity',
      'transition-duration': 140,
    },
  },
  {
    selector: 'node[type = "patient"]',
    css: {
      width: 92,
      height: 92,
      'font-size': 10,
      'font-weight': 800,
      'border-width': 2.2,
      'text-max-width': 58,
      'text-outline-width': 2,
    },
  },
  {
    selector: 'node:selected',
    css: {
      'border-color': '#ffffff',
      'border-width': 3.5,
    },
  },
  {
    selector: 'node.search-match',
    css: {
      'border-color': '#facc15',
      'border-width': 4,
    },
  },
  {
    selector: '.dimmed',
    css: {
      opacity: 0.22,
    },
  },
  {
    selector: 'edge',
    css: {
      width: 1.7,
      'line-color': '#94a3b8',
      'curve-style': 'bezier',
      opacity: 0.48,
      'overlay-opacity': 0,
    },
  },
  {
    selector: 'edge[type = "pmi_bundle"]',
    css: {
      width: 2.8,
      'line-color': '#22c55e',
      'line-style': 'dashed',
      opacity: 0.86,
    },
  },
  {
    selector: 'edge.search-match',
    css: {
      width: 3,
      opacity: 0.92,
    },
  },
] as unknown as StylesheetCSS[];

const GraphCanvas = forwardRef<GraphCanvasHandle, GraphCanvasProps>(
  ({ graph, searchTerm, onNodeSelect }, ref) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const cyRef = useRef<Core | null>(null);
    const [tooltip, setTooltip] = useState<TooltipState>({ visible: false, x: 0, y: 0 });

    const elements = useMemo<ElementDefinition[]>(() => {
      const nodes = graph.nodes.map((node) => {
        const type = normalizeType(node.type);
        const value = nodeValue(node);
        const importance = typeof node.importance === 'number' ? Math.abs(node.importance) : 0;
        const labelLength = String(node.label).length;
        return {
          data: {
            ...node,
            id: String(node.id),
            type,
            typeLabel: TYPE_LABELS[type] ?? type,
            displayLabel: displayLabel(node),
            color: TYPE_COLORS[type as keyof typeof TYPE_COLORS] ?? '#94a3b8',
            valueText: formatValue(value),
            fontSize: type === 'patient' ? 12 : labelLength > 15 ? 7 : labelLength > 11 ? 8 : 10,
            size: type === 'patient' ? 1 : Math.min(1, 0.24 + importance * 0.76),
            rawNode: node,
          },
        };
      });
      const edges = graph.edges
        .map((edge, index) => {
          const { source, target } = edgeEndpoints(edge);
          return {
            data: {
              ...edge,
              id: `edge-${index}-${source}-${target}`,
              source,
              target,
              type: edgeType(edge),
              pmiLabel: typeof edge.pmi === 'number' ? edge.pmi.toFixed(2) : '',
            },
          };
        })
        .filter((edge) => edge.data.source !== 'undefined' && edge.data.target !== 'undefined');
      return [...nodes, ...edges];
    }, [graph]);

    const applyLayout = (fit = false) => {
      const cy = cyRef.current;
      const container = containerRef.current;
      if (!cy || !container) return;
      const bounds = container.getBoundingClientRect();
      const positions = positionsForGraph(graph, bounds.width, bounds.height);
      cy.nodes().unlock();
      cy.nodes().forEach((node) => {
        const position = positions.get(node.id());
        if (position) node.position(position);
      });
      cy.layout({ name: 'preset', fit: false, animate: false }).run();
      cy.nodes().lock();
      if (fit) cy.fit(undefined, 46);
    };

    useImperativeHandle(ref, () => ({
      zoomIn: () => {
        const cy = cyRef.current;
        if (cy) cy.zoom({ level: Math.min(cy.zoom() * 1.18, cy.maxZoom()), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
      },
      zoomOut: () => {
        const cy = cyRef.current;
        if (cy) cy.zoom({ level: Math.max(cy.zoom() / 1.18, cy.minZoom()), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
      },
      reset: () => {
        applyLayout(true);
      },
      fit: () => {
        cyRef.current?.fit(undefined, 46);
      },
    }));

    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      const cy = cytoscape({
        container,
        elements,
        style: cytoscapeStyle,
        minZoom: 0.36,
        maxZoom: 2.4,
        boxSelectionEnabled: false,
        autoungrabify: true,
      });
      cyRef.current = cy;
      applyLayout(true);

      cy.on('mouseover', 'node', (event) => {
        const node = event.target as NodeSingular;
        const rawNode = node.data('rawNode') as PatientGraphNode;
        const rendered = node.renderedPosition();
        setTooltip({ visible: true, x: rendered.x, y: rendered.y, node: rawNode });
      });

      cy.on('mouseout', 'node', () => {
        setTooltip((current) => ({ ...current, visible: false }));
      });

      cy.on('tap', 'node', (event) => {
        const node = event.target as NodeSingular;
        onNodeSelect(node.data('rawNode') as PatientGraphNode);
      });

      cy.on('tap', (event) => {
        if (event.target === cy) onNodeSelect(null);
      });

      const observer = new ResizeObserver(() => {
        cy.resize();
        applyLayout(true);
      });
      observer.observe(container);

      return () => {
        observer.disconnect();
        cy.destroy();
        cyRef.current = null;
      };
    }, [elements, graph, onNodeSelect]);

    useEffect(() => {
      const cy = cyRef.current;
      if (!cy) return;
      const term = searchTerm.trim().toLowerCase();
      cy.elements().removeClass('search-match dimmed');
      if (!term) return;

      const matches = cy.nodes().filter((node) => {
        const label = String(node.data('label') ?? '').toLowerCase();
        const id = node.id().toLowerCase();
        const original = String(node.data('original_feature_name') ?? '').toLowerCase();
        return label.includes(term) || id.includes(term) || original.includes(term);
      });
      const connectedEdges = matches.connectedEdges();
      const connectedNodes = connectedEdges.connectedNodes();
      cy.elements().addClass('dimmed');
      matches.addClass('search-match').removeClass('dimmed');
      connectedNodes.removeClass('dimmed');
      connectedEdges.addClass('search-match').removeClass('dimmed');
      if (matches.length > 0) cy.animate({ fit: { eles: matches.union(connectedEdges), padding: 96 }, duration: 220 });
    }, [searchTerm]);

    const tooltipNode = tooltip.node;
    const tooltipType = tooltipNode ? normalizeType(tooltipNode.type) : '';

    return (
      <div className="graph-canvas-shell">
        <div ref={containerRef} className="graph-canvas" />
        {tooltip.visible && tooltipNode ? (
          <div className="node-tooltip" style={{ left: tooltip.x + 16, top: tooltip.y + 16 }}>
            <div className="tooltip-title">{tooltipNode.label}</div>
            <div className="tooltip-meta">{TYPE_LABELS[tooltipType] ?? tooltipType}</div>
            {formatValue(nodeValue(tooltipNode)) ? <div className="tooltip-meta">Value {formatValue(nodeValue(tooltipNode))}</div> : null}
            {tooltipNode.original_feature_name ? <div className="tooltip-meta">{tooltipNode.original_feature_name}</div> : null}
            {typeof tooltipNode.importance === 'number' ? <div className="tooltip-meta">Importance {tooltipNode.importance.toFixed(3)}</div> : null}
          </div>
        ) : null}
      </div>
    );
  },
);

export default GraphCanvas;
export { TYPE_COLORS, TYPE_LABELS, formatValue, nodeValue, normalizeType };
