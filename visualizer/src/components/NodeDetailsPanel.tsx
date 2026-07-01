import { AlertTriangle, X } from 'lucide-react';
import { getChiefComplaintInfo } from '../data/chiefComplaintDescriptions';
import { getMedicationInfo } from '../data/medicationDescriptions';
import { getVitalInfo } from '../data/vitalDescriptions';
import type { PatientGraphNode } from '../types/graph';
import { TYPE_COLORS, TYPE_LABELS, formatValue, nodeValue, normalizeType } from './GraphCanvas';

interface NodeDetailsPanelProps {
  node: PatientGraphNode;
  onClose: () => void;
}

const preferredKeys = [
  'id',
  'label',
  'type',
  'value',
  'z',
  'unit',
  'raw_value',
  'original_feature_name',
  'importance',
  'abnormal',
  'description',
];

function formatDetailValue(value: unknown) {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (value === null || value === undefined || value === '') return 'n/a';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function metricPercent(value: unknown) {
  if (typeof value !== 'number') return 0;
  return Math.max(4, Math.min(100, Math.abs(value) * 100));
}

export default function NodeDetailsPanel({ node, onClose }: NodeDetailsPanelProps) {
  const type = normalizeType(node.type);
  const color = TYPE_COLORS[type as keyof typeof TYPE_COLORS] ?? '#94a3b8';
  const value = nodeValue(node);
  const valueText = formatValue(value) || 'n/a';
  const importanceText = typeof node.importance === 'number' ? node.importance.toFixed(3) : 'n/a';
  const medicationInfo =
    type === 'medication' ? getMedicationInfo(node.label, node.original_feature_name) : null;
  const vitalInfo = type === 'vital' ? getVitalInfo(node) : null;
  const chiefComplaintInfo = type === 'chief_complaint' ? getChiefComplaintInfo(node) : null;
  const keys = [
    ...preferredKeys.filter((key) => Object.prototype.hasOwnProperty.call(node, key)),
    ...Object.keys(node).filter((key) => !preferredKeys.includes(key)),
  ];

  return (
    <aside className="side-panel">
      <div className="panel-accent" style={{ background: color }} />

      <div className="panel-header">
        <div>
          <p className="panel-kicker">
            <span className="dot" style={{ background: color }} />
            <span>{TYPE_LABELS[type] ?? type}</span>
          </p>
          <h2>{node.label}</h2>
        </div>
        <button className="panel-close" type="button" onClick={onClose} aria-label="Close details">
          <X size={17} />
        </button>
      </div>

      <div className="panel-body">
        <div className="metric-grid">
          <div className="metric-card">
            <div className="metric-label">Value</div>
            <div className="metric-value">{valueText}</div>
            <div className="metric-sub">{node.unit ? String(node.unit) : 'normalized feature'}</div>
            <div className="metric-bar">
              <i style={{ width: `${metricPercent(typeof value === 'number' ? value / 2.4 : 0)}%` }} />
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Importance</div>
            <div className="metric-value">{importanceText}</div>
            <div className="metric-sub">model signal</div>
            <div className="metric-bar">
              <i style={{ width: `${metricPercent(node.importance)}%` }} />
            </div>
          </div>
        </div>

        {node.abnormal ? (
          <div className="abnormal-badge">
            <span className="dot" />
            <AlertTriangle size={14} />
            <span>Abnormal feature flag</span>
          </div>
        ) : null}

        {node.description || node.original_feature_name ? (
          <section className="panel-desc">
            <div className="panel-section-label">Clinical feature</div>
            <p>{String(node.description ?? node.original_feature_name)}</p>
          </section>
        ) : null}

        <dl className="detail-list">
          <div className="panel-section-label">All details</div>
          {keys.map((key) => (
            <div key={key} className="detail-row">
              <dt>{key.replace(/_/g, ' ')}</dt>
              <dd>{formatDetailValue(node[key])}</dd>
            </div>
          ))}
        </dl>

        {medicationInfo ? (
          <section className="medication-info-card">
            <div className="panel-section-label">Medication description</div>
            <h3>{medicationInfo.className}</h3>
            <p>{medicationInfo.description}</p>
          </section>
        ) : null}

        {vitalInfo ? (
          <section className="vital-info-card">
            <div className="panel-section-label">Vital sign description</div>
            <h3>{vitalInfo.displayName}</h3>
            <p>{vitalInfo.description}</p>
          </section>
        ) : null}

        {chiefComplaintInfo ? (
          <section className="chief-complaint-info-card">
            <div className="panel-section-label">Chief complaint description</div>
            <h3>{chiefComplaintInfo.displayName}</h3>
            <p>{chiefComplaintInfo.description}</p>
          </section>
        ) : null}
      </div>
    </aside>
  );
}
