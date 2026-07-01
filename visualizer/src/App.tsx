import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Maximize2, Minus, Plus, RotateCcw, Search } from 'lucide-react';
import GraphCanvas, { GraphCanvasHandle, normalizeType } from './components/GraphCanvas';
import Legend from './components/Legend';
import NodeDetailsPanel from './components/NodeDetailsPanel';
import { getDiagnosisInfo } from './data/diagnosisDescriptions';
import { loadGraph, loadManifest } from './data/graphLoader';
import type { GraphManifestItem, PatientGraph, PatientGraphNode } from './types/graph';

function graphId(graph: PatientGraph) {
  return graph.graph_id ?? graph.index ?? 'unknown';
}

function patientLabelOf(graph: PatientGraph): string | null {
  if (graph.subject_id !== undefined && graph.subject_id !== null) return String(graph.subject_id);
  const patient = graph.nodes.find((node) => normalizeType(node.type) === 'patient');
  if (!patient) return graphId(graph) === 'unknown' ? null : String(graphId(graph));
  const label = String(patient.label).trim();
  if (!label || label.toUpperCase() === 'PATIENT') return String(graphId(graph));
  return label;
}

interface GraphOption {
  file: string;
  label: string;
}

function App() {
  const graphRef = useRef<GraphCanvasHandle | null>(null);
  const cacheRef = useRef<Record<string, PatientGraph>>({});

  const [manifest, setManifest] = useState<GraphManifestItem[]>([]);
  const [activeFile, setActiveFile] = useState('');
  const [graph, setGraph] = useState<PatientGraph | null>(null);
  const [selectedNode, setSelectedNode] = useState<PatientGraphNode | null>(null);
  const [graphQuery, setGraphQuery] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [status, setStatus] = useState('Loading patient graphs');

  // Load only the manifest up front. Individual graph shards are fetched lazily
  // when the user selects a graph, which keeps the full dataset usable in-browser.
  useEffect(() => {
    let cancelled = false;
    loadManifest()
      .then((items) => {
        if (cancelled) return;
        setManifest(items);
        setActiveFile((current) => current || items[0]?.file || '');
      })
      .catch((error: Error) => {
        if (!cancelled) setStatus(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Resolve the active graph — from cache when available, otherwise fetch.
  useEffect(() => {
    if (!activeFile) return;
    let cancelled = false;
    setSelectedNode(null);

    const cached = cacheRef.current[activeFile];
    if (cached) {
      setGraph(cached);
      setStatus('');
      return;
    }

    setStatus('Loading patient graph');
    loadGraph(activeFile)
      .then((nextGraph) => {
        if (cancelled) return;
        cacheRef.current[activeFile] = nextGraph;
        setGraph(nextGraph);
        setStatus('');
      })
      .catch((error: Error) => {
        if (!cancelled) setStatus(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [activeFile]);

  const handleNodeSelect = useCallback((node: PatientGraphNode | null) => {
    setSelectedNode(node);
  }, []);

  const dropdownOptions: GraphOption[] = useMemo(() => {
    const query = graphQuery.trim().toLowerCase();
    const matches = query
      ? manifest.filter((item) => {
          const haystack = [
            item.label,
            item.graph_id,
            item.subject_id,
            item.diagnosis,
          ].join(' ').toLowerCase();
          return haystack.includes(query);
        })
      : manifest;

    const visible = matches.slice(0, 350).map((item) => ({ file: item.file, label: item.label }));
    const activeItem = manifest.find((item) => item.file === activeFile);
    if (activeItem && !visible.some((item) => item.file === activeItem.file)) {
      visible.push({ file: activeItem.file, label: activeItem.label });
    }
    return visible;
  }, [activeFile, graphQuery, manifest]);

  const activePatientLabel = graph ? patientLabelOf(graph) : null;
  const diagnosisInfo = getDiagnosisInfo(graph?.diagnosis);
  const nodeCount = graph?.nodes.length ?? 0;
  const pmiEdgeCount = useMemo(
    () =>
      graph?.edges.filter((edge) => {
        const type = edge.type ?? edge.kind;
        return type === 'pmi_bundle' || type === 'pmi';
      }).length ?? 0,
    [graph],
  );

  return (
    <main className="app-shell">
      <div className="graph-layer">
        <div className="graph-grid" aria-hidden="true" />
        {graph ? (
          <GraphCanvas ref={graphRef} graph={graph} searchTerm={searchTerm} onNodeSelect={handleNodeSelect} />
        ) : null}
        {status ? (
          <div className="loading-state">
            <div className="loading-inner">
              <div className="spinner" />
              <div className="loading-label">{status}</div>
            </div>
          </div>
        ) : null}
      </div>

      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12h4l2.5-7 4.5 14 2.5-7H21" />
            </svg>
          </div>
          <div className="brand-text">
            <p className="eyebrow">
              ProtGNN · {activePatientLabel ? `Patient ${activePatientLabel}` : graph ? `Graph ${graphId(graph)}` : '····'}
            </p>
            <h1>{graph?.diagnosis ?? 'Loading patient graph'}</h1>
          </div>
        </div>

        <div className="toolbar">
          <label className="search-box graph-search-box" htmlFor="graph-search">
            <Search size={16} />
            <input
              id="graph-search"
              type="search"
              placeholder="Find patient"
              value={graphQuery}
              onChange={(event) => setGraphQuery(event.target.value)}
            />
          </label>

          <div className="select-wrap">
            <select
              id="graph-select"
              aria-label="Select patient"
              value={activeFile}
              onChange={(event) => setActiveFile(event.target.value)}
              disabled={dropdownOptions.length === 0}
            >
              {dropdownOptions.map((option) => (
                <option key={option.file} value={option.file}>
                  {option.label}
                </option>
              ))}
            </select>
            <span className="select-caret" aria-hidden="true">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="m6 9 6 6 6-6" />
              </svg>
            </span>
          </div>

          <label className="search-box" htmlFor="node-search">
            <Search size={16} />
            <input
              id="node-search"
              type="search"
              placeholder="Search nodes"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
            />
          </label>

          <div className="toolbar-divider" aria-hidden="true" />

          <div className="stat-chip">
            <b>{nodeCount}</b>
            <span>nodes</span>
          </div>
          <div className="stat-chip">
            <span className="stat-dot" aria-hidden="true" />
            <b>{pmiEdgeCount}</b>
            <span>PMI</span>
          </div>
        </div>
      </header>

      <section className="diagnosis-card" aria-label="Diagnosis description">
        <p className="diagnosis-card-kicker">Diagnosis</p>
        <h2>{graph?.diagnosis ?? 'Loading patient graph'}</h2>
        <p className="diagnosis-card-description">{diagnosisInfo.description}</p>
        {diagnosisInfo.mergedFrom?.length ? (
          <div className="diagnosis-merge-box">
            <span>Grouped under this diagnosis</span>
            <ul className="diagnosis-merge-list">
              {diagnosisInfo.mergedFrom.map((label) => (
                <li key={label}>{label}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <div className="legend-dock">
        <Legend />
      </div>

      <div className="control-dock" aria-label="Graph viewport controls">
        <button className="icon-button" type="button" onClick={() => graphRef.current?.zoomIn()} aria-label="Zoom in" title="Zoom in">
          <Plus size={18} />
        </button>
        <button className="icon-button" type="button" onClick={() => graphRef.current?.zoomOut()} aria-label="Zoom out" title="Zoom out">
          <Minus size={18} />
        </button>
        <div className="control-sep" aria-hidden="true" />
        <button className="icon-button" type="button" onClick={() => graphRef.current?.fit()} aria-label="Fit to screen" title="Fit to screen">
          <Maximize2 size={17} />
        </button>
        <button className="icon-button" type="button" onClick={() => graphRef.current?.reset()} aria-label="Reset view" title="Reset view">
          <RotateCcw size={16} />
        </button>
      </div>

      {selectedNode ? <NodeDetailsPanel node={selectedNode} onClose={() => setSelectedNode(null)} /> : null}
    </main>
  );
}

export default App;
