import { TYPE_COLORS } from './GraphCanvas';

const legendItems = [
  {
    label: 'Patient',
    color: TYPE_COLORS.patient,
    description: 'The single ED visit (stay_id) at the graph center.',
  },
  {
    label: 'Vital sign',
    color: TYPE_COLORS.vital,
    description: 'Triage and visit vital measurements.',
  },
  {
    label: 'Medication',
    color: TYPE_COLORS.medication,
    description: 'Drugs administered or prescribed.',
  },
  {
    label: 'Chief complaint',
    color: TYPE_COLORS.chief_complaint,
    description: "The patient's main triage complaint category.",
  },
];

const medicationPmiDescription =
  'A co-occurrence edge drawn between two drugs whose pointwise mutual information exceeds 2.0, about 7.4x more often together than chance.';

export default function Legend() {
  return (
    <div className="legend" aria-label="Node legend">
      {legendItems.map(({ label, color, description }) => (
        <div className="legend-item" key={label} aria-label={`${label}: ${description}`} title={description} tabIndex={0}>
          <span className="legend-dot" style={{ background: color }} />
          <span>{label}</span>
          <span className="legend-tooltip" role="tooltip">
            {description}
          </span>
        </div>
      ))}
      <div
        className="legend-edge"
        aria-label={`Medication PMI: ${medicationPmiDescription}`}
        title={medicationPmiDescription}
        tabIndex={0}
      >
        <span className="legend-dash" />
        <span>Medication PMI</span>
        <span className="legend-tooltip" role="tooltip">
          {medicationPmiDescription}
        </span>
      </div>
    </div>
  );
}
