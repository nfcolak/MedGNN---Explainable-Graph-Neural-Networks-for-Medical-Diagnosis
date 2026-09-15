import { Sparkles, X } from 'lucide-react';
import type { PatientGraph, PatientGraphNode } from '../types/graph';
import { TYPE_COLORS, TYPE_LABELS, formatValue, nodeValue, normalizeType } from './GraphCanvas';

interface EvidencePanelProps {
  graph: PatientGraph;
  onSelect: (node: PatientGraphNode) => void;
  onClose: () => void;
}

function heatColor(t: number) {
  const clamped = Math.max(0, Math.min(1, t));
  const from = [252, 211, 77];
  const to = [240, 101, 74];
  const channel = (i: number) => Math.round(from[i] + (to[i] - from[i]) * clamped);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

export default function EvidencePanel({ graph, onSelect, onClose }: EvidencePanelProps) {
  const scored = graph.nodes
    .filter((node) => typeof node.importance === 'number' && normalizeType(node.type) !== 'patient')
    .map((node) => ({ node, importance: Math.abs(node.importance as number) }))
    .sort((a, b) => b.importance - a.importance);

  const hasScores = scored.length > 0;
  const max = hasScores ? scored[0].importance : 0;
  const ranked = scored.slice(0, 8);

  return (
    <aside className="side-panel evidence-panel">
      <div className="panel-accent" style={{ background: 'linear-gradient(90deg, #fcd34d, #f0654a)' }} />

      <div className="panel-header">
        <div>
          <p className="panel-kicker">
            <Sparkles size={13} color="#f5a524" />
            <span>Explanation mode</span>
          </p>
          <h2>Model evidence</h2>
        </div>
        <button className="panel-close" type="button" onClick={onClose} aria-label="Close explanation panel">
          <X size={17} />
        </button>
      </div>

      <div className="panel-body">
        {hasScores ? (
          <>
            <p className="evidence-intro">
              Nodes ranked by attribution — how strongly each feature drove the model's prediction of{' '}
              <b>{graph.diagnosis}</b>. Brighter and larger nodes on the graph carry more weight.
            </p>
            <ol className="evidence-list">
              {ranked.map(({ node, importance }, index) => {
                const type = normalizeType(node.type);
                const norm = max > 0 ? importance / max : 0;
                const valueText = formatValue(nodeValue(node));
                return (
                  <li key={String(node.id)}>
                    <button className="evidence-row" type="button" onClick={() => onSelect(node)}>
                      <span className="evidence-rank">{index + 1}</span>
                      <span className="evidence-main">
                        <span className="evidence-name">
                          <span
                            className="evidence-type-dot"
                            style={{ background: TYPE_COLORS[type as keyof typeof TYPE_COLORS] ?? '#94a3b8' }}
                          />
                          {node.label}
                          {valueText ? <em>{valueText}</em> : null}
                        </span>
                        <span className="evidence-track">
                          <span
                            className="evidence-fill"
                            style={{ width: `${Math.max(6, norm * 100)}%`, background: heatColor(norm) }}
                          />
                        </span>
                      </span>
                      <span className="evidence-score">{importance.toFixed(3)}</span>
                    </button>
                    <span className="evidence-typelabel">{TYPE_LABELS[type] ?? type}</span>
                  </li>
                );
              })}
            </ol>
          </>
        ) : (
          <div className="evidence-empty">
            <Sparkles size={22} color="#6f675d" />
            <p>No attribution scores were exported for this patient.</p>
            <p className="evidence-empty-sub">
              Pick an <b>★ Explained example</b> from the patient selector to see the model's evidence overlay.
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}
