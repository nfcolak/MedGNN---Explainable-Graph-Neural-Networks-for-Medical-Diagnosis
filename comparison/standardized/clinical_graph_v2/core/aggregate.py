"""Descriptive aggregation of recorded, binding-matched clinical experiment arms.

Seed SD is optimization/subsampling variability, not a significance threshold or
patient-cluster uncertainty. Never select a winner across runs or cohorts. Missing
bindings remain explicit; legacy results are displayed but not silently certified.
Parameter-count groups, standalone/tabular rows and HGT relation-separation
measurements are retained independently of comparison eligibility.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

LEGACY = 'legacy_unspecified'
# These are intentional arm interventions, not permission to mix input universes.
TREATMENT_FIELDS = {'method', 'adaptation_version', 'method_config',
                    'method_native_defaults', 'common_overrides', 'method_overrides',
                    'conv', 'edges', 'edge_payload', 'message_passing',
                    'dropped_relations', 'rewired_relations', 'rewire_seed',
                    'rewiring_policy', 'active_parameter_count', 'parameter_count',
                    'modulation', 'heads', 'hidden', 'layers', 'dropout', 'lr',
                    'weight_decay', 'batch_size', 'epochs', 'patience', 'min_delta',
                    'early_stopping_start_epoch_index', 'optimizer', 'weights',
                    # Edge view and GCHM-PNA v2 switches are arm interventions like
                    # `edges`; the derived widths/vocabularies follow from them.
                    'edge_direction', 'num_relations', 'edge_feature_layout', 'edge_dim',
                    'num_meta_relations', 'meta_relations', 'aggregation', 'readout',
                    'hub_gate', 'max_depth', 'learning_rate', 'round_step',
                    # GCHM-PNA v3 switches.
                    'v3_readout', 'wide', 'jk', 'edge_dropout'}
# Realized null outcomes may vary per seed; retain them in each run, not the
# cohort key or the fixed treatment settings. They are not tuning parameters.
OUTCOME_FIELDS = {'rewiring', 'selected_validation', 'selected_dev', 'selected_rounds'}
LOCATION_FIELDS = {'artifact', 'targets', 'out', 'output', 'matched_against'}
VERSION_FIELDS = ('logic_contract_version', 'evaluation_version', 'preprocessing_schema_version')
TOP_LEVEL_BINDINGS = (
    'method', 'features', 'rounds', 'seed', 'counts', 'num_classes', 'class_order', 'label_order',
    'top_k_labels', 'kept_label_indices', 'min_prior_visits', 'train_limit',
    'test_evaluated', 'evaluation_fold', 'selection', 'artifact_graphs_sha256',
    'artifact_visit_membership_sha256', 'visit_membership_contract_version',
    'targets_sha256', 'target_binding_sha256', 'source_code',
    'split_sample_ids_sha256', 'feature_contract_sha256', 'preprocessing_sha256',
    'input_contract_version',
    *VERSION_FIELDS,
)


def split_name(name):
    """Only a literal numeric _seed suffix is a seed binding; no inferred defaults."""
    match = re.fullmatch(r'(.+)_seed([0-9]+)', name)
    return (match[1], match[2]) if match else (name, None)


def binding(row):
    out = {k: row[k] for k in TOP_LEVEL_BINDINGS if k in row}
    for k, value in row.get('binding', {}).items():
        if k in out and out[k] != value:
            raise ValueError('Conflicting top-level/binding field: ' + k)
        out[k] = value
    return out


def _contract(row):
    b = binding(row)
    contract = {k: v for k, v in b.items()
                if k not in TREATMENT_FIELDS | LOCATION_FIELDS | OUTCOME_FIELDS | {'seed'}}
    for field in VERSION_FIELDS:
        contract.setdefault(field, LEGACY)
    if 'class_order' not in contract and contract.get('label_order'):
        contract['class_order'] = contract['label_order']
    # Class order is recorded by per_class on older runs, not inferred from a
    # hard-coded class count (which used to be 30 even on reduced-class tasks).
    if 'class_order' not in contract and row.get('per_class'):
        table = row['per_class']
        if all('index' in entry for entry in table):
            table = sorted(table, key=lambda entry: entry['index'])
            if [entry['index'] for entry in table] != list(range(len(table))):
                raise ValueError('Invalid per-class indices')
        contract['class_order'] = [entry['label'] for entry in table]
    labels = contract.get('class_order')
    if labels is not None:
        if (not isinstance(labels, list) or not labels or len(labels) != len(set(labels))
                or contract.get('num_classes', len(labels)) != len(labels)
                or ('label_order' in contract and contract['label_order'] != labels)):
            raise ValueError('Contradictory or invalid class universe')
    return contract


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None


def _missing_bindings(contract, full_bindings=None):
    if full_bindings is None:
        full_bindings = [contract]
    elif isinstance(full_bindings, dict):
        full_bindings = [full_bindings]
    missing = []
    for key in ('artifact_graphs_sha256', 'artifact_visit_membership_sha256',
                'targets_sha256', 'target_binding_sha256'):
        if not _digest(contract.get(key)):
            missing.append(key)
    if contract.get('visit_membership_contract_version') in (None, LEGACY):
        missing.append('visit_membership_contract_version')
    split_hashes = contract.get('split_sample_ids_sha256')
    if (not isinstance(split_hashes, dict)
            or not all(_digest(split_hashes.get(f)) for f in ('train', 'validation'))):
        missing.append('split_sample_ids_sha256.train_and_validation')
    sources = contract.get('source_code')
    if not isinstance(sources, dict) or not sources or not all(_digest(v) for v in sources.values()):
        missing.append('source_code')
    if not contract.get('class_order'):
        missing.append('class_order')
    counts = contract.get('counts', {})
    if not all(isinstance(counts.get(f), int) and counts[f] > 0 for f in ('train', 'validation')):
        missing.append('counts.train_and_validation')
    if (not full_bindings
            or any(not any(binding.get(k) is not None
                           for k in ('epochs', 'rounds', 'training_budget'))
                   for binding in full_bindings)):
        missing.append('training_budget')
    if not contract.get('selection'):
        missing.append('selection')
    if (contract.get('preprocessing_schema_version') in (None, LEGACY)
            and not any(_digest(contract.get(k)) for k in ('feature_contract_sha256', 'preprocessing_sha256'))):
        missing.append('preprocessing_schema_version_or_feature_hash')
    for field in ('logic_contract_version', 'evaluation_version'):
        if contract.get(field) in (None, LEGACY):
            missing.append(field)
    return missing


def _summary(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _metrics(row):
    # Dev-selected tuning runs never read validation and record metrics=None.
    metrics = row.get('metrics') or {}
    if 'macro_f1' not in metrics:
        raise ValueError('Run is missing macro_f1 (tuning runs without validation '
                         'metrics are not aggregated)')
    for metric, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('Nonfinite or nonnumeric metric: ' + metric)
    return metrics


def _selected_validation_issues(row, full_binding, metrics):
    """Validate checkpoint evidence without adding seed outcomes to cohort identity."""
    selected = full_binding.get('selected_validation')
    expected_keys = {
        'epoch', 'epoch_index', 'metric', 'metric_value',
        'prediction_sha256', 'sample_ids_sha256',
    }
    if not isinstance(selected, dict) or set(selected) != expected_keys:
        return ['selected_validation.missing_or_malformed']
    issues = []
    epoch = selected.get('epoch')
    epoch_index = selected.get('epoch_index')
    if (type(epoch) is not int or epoch < 1 or type(epoch_index) is not int
            or epoch_index < 0 or epoch_index + 1 != epoch
            or row.get('selected_epoch') != epoch):
        issues.append('selected_validation.epoch')
    metric_value = selected.get('metric_value')
    if (selected.get('metric') != 'macro_f1'
            or isinstance(metric_value, bool)
            or not isinstance(metric_value, (float, int))
            or not math.isfinite(metric_value)
            or not math.isclose(float(metric_value), float(metrics['macro_f1']),
                                rel_tol=0.0, abs_tol=1e-12)):
        issues.append('selected_validation.metric')
    prediction_hash = selected.get('prediction_sha256')
    if (not _digest(prediction_hash) or prediction_hash != row.get('proba_sha256')):
        issues.append('selected_validation.prediction_sha256')
    validation_ids = full_binding.get('split_sample_ids_sha256', {}).get('validation')
    if (not _digest(selected.get('sample_ids_sha256'))
            or selected.get('sample_ids_sha256') != validation_ids):
        issues.append('selected_validation.sample_ids_sha256')
    return issues


def aggregate_rows(rows, base='nomp'):
    """Partition by ALL non-treatment bindings, including hashes and full budget.

    Unrecognized binding fields are retained in the key (fail closed against
    future schema changes). Matching names/counts alone never prove parity.
    """
    if not rows:
        raise ValueError('No result rows')
    buckets = defaultdict(list)
    runs = {}
    for name, row in sorted(rows.items()):
        if row.get('status', 'completed') != 'completed':
            raise ValueError('Run is not completed: ' + name)
        b = binding(row)
        metrics = _metrics(row)
        validation_issues = _selected_validation_issues(row, b, metrics)
        arm, suffix_seed = split_name(name)
        stored_seed = str(b['seed']) if b.get('seed') is not None else None
        if suffix_seed is not None and stored_seed is not None and suffix_seed != stored_seed:
            raise ValueError('Directory seed disagrees with bound seed: ' + name)
        seed = stored_seed or suffix_seed
        contract = _contract(row)
        run = {'name': name, 'arm': arm, 'seed': seed, 'seed_is_bound': stored_seed is not None, 'binding': b,
               'metrics': metrics, 'selected_epoch': row.get('selected_epoch'),
               'binding_issues': validation_issues,
               'relation_separation': row.get('relation_separation'),
               'per_class': row.get('per_class'),
               'majority_baseline_accuracy': row.get('majority_baseline_accuracy')}
        runs[name] = run
        buckets[_key(contract)].append(run)
    cohorts = []
    for cohort_key, members in sorted(buckets.items()):
        contract = json.loads(cohort_key)
        by_arm = defaultdict(list)
        for run in members:
            by_arm[run['arm']].append(run)
        arms, seed_maps = {}, {}
        for arm, arm_runs in sorted(by_arm.items()):
            settings = {_key({k: v for k, v in r['binding'].items()
                              if k in TREATMENT_FIELDS - {'rewire_seed'}}) for r in arm_runs}
            if len(settings) > 1:
                raise ValueError('Inconsistent treatment settings within arm: ' + arm)
            seeds = [r['seed'] for r in arm_runs if r['seed'] is not None]
            if len(seeds) != len(set(seeds)):
                raise ValueError('Duplicate seed within arm/cohort: ' + arm)
            seed_maps[arm] = {r['seed']: r for r in arm_runs if r['seed_is_bound']}
            metric_names = sorted(set().union(*(r['metrics'] for r in arm_runs)))
            arms[arm] = {'n_runs': len(arm_runs), 'n_seeds': len(seeds), 'seeds': sorted(seeds),
                         'unbound_seed_runs': [r['name'] for r in arm_runs if not r['seed_is_bound']],
                         'unseeded_runs': [r['name'] for r in arm_runs if r['seed'] is None],
                         'runs': [r['name'] for r in arm_runs],
                         'treatment': json.loads(next(iter(settings))),
                         'metrics': {m: _summary(r['metrics'][m] for r in arm_runs if m in r['metrics'])
                                     for m in metric_names}}
        missing = _missing_bindings(
            contract, [member['binding'] for member in members])
        missing.extend(
            f"{member['name']}:{issue}"
            for member in members for issue in member['binding_issues'])
        missing = sorted(set(missing))
        comparisons = []
        ineligible_comparisons = []
        if not missing and base in arms:
            ref = arms[base]
            for arm, summary in sorted(arms.items()):
                if arm == base:
                    continue
                if summary['treatment'].get('rewired_relations') and (
                        ref['treatment'].get('edge_payload') is not False
                        or summary['treatment'].get('edge_payload') is not False):
                    ineligible_comparisons.append({
                        'arm': arm, 'reference': base,
                        'reason': 'rewiring requires edge_payload=false in both arms'})
                    continue
                matched = sorted(seed_maps[base].keys() & seed_maps[arm].keys())
                common_metrics = sorted(ref['metrics'].keys() & summary['metrics'].keys())
                paired = {m: _summary(seed_maps[arm][s]['metrics'][m] - seed_maps[base][s]['metrics'][m]
                                      for s in matched if m in seed_maps[arm][s]['metrics']
                                      and m in seed_maps[base][s]['metrics']) for m in common_metrics}
                delta = summary['metrics']['macro_f1']['mean'] - ref['metrics']['macro_f1']['mean']
                ref_sd = ref['metrics']['macro_f1']['sample_sd']
                comparisons.append({
                    'arm': arm, 'reference': base,
                    'status': 'binding_matched_descriptive_only',
                    'paired_seeds': matched, 'n_pairs': len(matched),
                    'unpaired_arm_seeds': sorted(seed_maps[arm].keys() - seed_maps[base].keys()),
                    'unpaired_reference_seeds': sorted(seed_maps[base].keys() - seed_maps[arm].keys()),
                    'paired_metrics': paired,
                    'unpaired_all_run_mean_delta': delta,
                    'descriptive_delta_over_reference_sd': delta / ref_sd if ref_sd else None,
                    'ratio_interpretation': 'Descriptive scaling only; not a significance test or decision threshold.',
                    'treatment_differences': {k: {'reference': ref['treatment'].get(k),
                                                'arm': summary['treatment'].get(k)}
                                              for k in sorted(ref['treatment'].keys() | summary['treatment'].keys())
                                              if ref['treatment'].get(k) != summary['treatment'].get(k)},
                })
        if ineligible_comparisons:
            comparisons = []
        cohorts.append({'id': len(cohorts) + 1, 'contract': contract, 'arms': arms,
                        'comparison_eligible': not missing and not ineligible_comparisons,
                        'missing_bindings': missing,
                        'ineligible_comparisons': ineligible_comparisons,
                        'comparisons': comparisons})
    return {
        'n_runs': len(runs), 'n_cohorts': len(cohorts), 'reference_arm': base,
        'cohorts': cohorts, 'runs': runs,
        'patient_cluster_uncertainty': {
            'status': 'not_measured',
            'reason': 'Seed replicates do not estimate patient-cluster sampling uncertainty; no patient bootstrap was run.'},
        'limitations': [
            'All statistics are descriptive; no significance threshold and no automatic winner.',
            'Cohorts match recorded bindings, not independently rehashed datasets or per-patient predictions.',
            'Legacy or incomplete bindings are displayed separately from eligible comparisons.',
            'Paired deltas use only shared seeds within a binding-matched cohort; unpaired means use all stated runs.',
            'Seed variation may combine initialization and seeded subsampling; a single run has no measured seed SD.',
            'Validation selected checkpoints have selection bias, not held-out-test evidence.',
            'Parameter count is a capacity descriptor, not a proof of equal compute or architecture equivalence.',
            'Fitted GNN node views and summarized tabular controls are not assumed information-equivalent.',
        ],
    }


def _number(value):
    return '-' if value is None else f'{value:.4f}'


def render_report(report, detail_run=None):
    """Keep every run visible; class diagnostics require an explicit run choice."""
    lines = ['=== TEK KOSU SONUCLARI (kayitli metrikler; yeni test degerlendirmesi yok) ===',
             f"{'run':28} {'seed':>6} {'conv/method':>18} {'hid':>5} {'param/feat':>10} {'active':>10} "
             f"{'ntrain':>8} {'nval':>8} {'ep':>4} {'acc':>8} {'bal':>8} {'mF1':>8} {'top3':>8}"]
    capacities = defaultdict(list)
    for name, run in report['runs'].items():
        b = run['binding']
        counts = b.get('counts', {})
        capacity = b.get('parameter_count', b.get('features'))
        method_name = b.get('conv') or b.get('method', LEGACY)
        lines.append(f"{name:28} {str(run['seed']):>6} {method_name:>18} "
                     f"{str(b.get('hidden')):>5} {str(capacity):>10} {str(b.get('active_parameter_count')):>10} "
                     f"{str(counts.get('train')):>8} {str(counts.get('validation')):>8} "
                     f"{str(run['selected_epoch']):>4} "
                     f"{_number(run['metrics'].get('accuracy')):>8} "
                     f"{_number(run['metrics'].get('balanced_acc')):>8} "
                     f"{_number(run['metrics']['macro_f1']):>8} "
                     f"{_number(run['metrics'].get('top3_acc')):>8}")
        capacities[b.get('parameter_count')].append(name)
    lines.append('\n=== KAPASITE GRUPLARI (parametre sayisi; esit compute kaniti degil) ===')
    for capacity, names in sorted(capacities.items(), key=lambda item: str(item[0])):
        lines.append(f'  {capacity}: {", ".join(names)}')
    for cohort in report['cohorts']:
        c = cohort['contract']
        lines.append(f"\n=== BAGLANTI KOHORTU {cohort['id']} ===")
        lines.append('  contract: ' + json.dumps(c, sort_keys=True, ensure_ascii=False))
        if not cohort['comparison_eligible']:
            if cohort['missing_bindings']:
                lines.append('  Karsilastirma kapali; eksik baglar: '
                             + ', '.join(cohort['missing_bindings']))
            for item in cohort.get('ineligible_comparisons', []):
                lines.append(f"  Karsilastirma dislandi: {item['arm']} vs {item['reference']}: "
                             f"{item['reason']}")
        for arm, s in cohort['arms'].items():
            m = s['metrics']['macro_f1']
            lines.append(f"  {arm}: n_runs={s['n_runs']} n_seeds={s['n_seeds']} seeds={s['seeds']} "
                         f"mF1 ort={_number(m['mean'])} SS={_number(m['sample_sd'])} "
                         f"min={_number(m['min'])} max={_number(m['max'])}")
        for comparison in cohort['comparisons']:
            p = comparison['paired_metrics']['macro_f1']
            lines.append(f"  {comparison['arm']} - {comparison['reference']}: "
                         f"eslesen seeds={comparison['paired_seeds']} n={p['n']} "
                         f"delta ort={_number(p['mean'])} delta SS={_number(p['sample_sd'])}; "
                         f"tum kosu ortalama farki={_number(comparison['unpaired_all_run_mean_delta'])}")
            lines.append(f"    eslesmeyen arm/reference seeds: {comparison['unpaired_arm_seeds']} / "
                         f"{comparison['unpaired_reference_seeds']}")
    typed = {n: r for n, r in report['runs'].items() if r['binding'].get('conv') == 'hgt'}
    if typed:
        lines.append('\n=== BG-HGNN ILISKI AYRISMASI (kayitli tanisal olcum) ===')
        for name, run in typed.items():
            sep = run['relation_separation']
            if not sep:
                lines.append(f'  {name}: olculmedi')
            else:
                lines.append(f"  {name}: k_rel init={[x['k_rel'] for x in sep.get('at_init', [])]} "
                             f"end={[x['k_rel'] for x in sep.get('at_end', [])]}")
    if detail_run:
        if detail_run not in report['runs']:
            raise ValueError('Unknown detail run: ' + detail_run)
        run = report['runs'][detail_run]
        pc = run['per_class'] or []
        lines.append(f'\n=== SECILEN KOSU SINIFLARI: {detail_run} (otomatik secim yok) ===')
        lines.append(f"  F1=0: {sum(c['f1'] == 0 for c in pc)}/{len(pc)}; "
                     f"cogunluk accuracy={_number(run['majority_baseline_accuracy'])}")
        for row in pc:
            lines.append(f"  {row['label']}: f1={_number(row['f1'])} "
                         f"recall={_number(row['recall'])} n={row['support']}")
    lines.extend(['\nHasta-kume belirsizligi: OLCULMEDI.',
                  'Seed SS ve oranlar yalniz betimseldir; anlamlilik esigi veya otomatik kazanan yok.'])
    lines.extend('  - ' + limitation for limitation in report['limitations'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='?', type=Path,
                        default=Path('comparison/standardized/clinical_runs_v3'))
    parser.add_argument('--reference', default='nomp')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--detail-run', default=None)
    args = parser.parse_args()
    rows = {}
    for path in sorted(args.runs.glob('*/result.json')):
        with path.open() as stream:
            rows[path.parent.name] = json.load(stream)
    if not rows:
        parser.error(f'no result.json under {args.runs}')
    report = aggregate_rows(rows, args.reference)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) if args.json
          else render_report(report, args.detail_run))


if __name__ == '__main__':
    main()
