"""Bounded regressions against the real builder; no training or raw extraction."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from core import audit
from core.schema import timestamp


class TinyStore:
    def visit(self, stay):
        return {'subject': '1', 'start': '2020-01-03T00:00:00',
                'finish': '2020-01-03T02:00:00'}

    def arrival(self, stay):
        return {}

    def complaints(self, stay):
        return ['pain', 'fever']

    def triage_vitals(self, stay):
        return []

    def measurements(self, stay, cutoff):
        return []

    def prior_visits(self, subject, start):
        return []


def tiny_graph():
    from core.graph import build_graph
    return build_graph(TinyStore(), {'stay_id': '1', 'subject_id': '1',
                       'cutoff': timestamp('2020-01-03T01:00:00'),
                       'sample_id': 'synthetic', 'split': 'train'}, [], {'pain', 'fever'})


def audit_graphs(graphs):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'graphs.jsonl'
        path.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
        return audit.audit(path)


class HistoryStore(TinyStore):
    def visit(self, stay):
        if stay == 'old':
            return {'subject': '1', 'start': '2020-01-01T00:00:00',
                    'finish': '2020-01-01T12:00:00'}
        return super().visit(stay)

    def prior_visits(self, subject, start):
        return ['old']

    @staticmethod
    def record(identifier, time, value, unit='mg/dL'):
        return {'id': identifier, 'source': identifier, 'token': 'lab:1',
                'time': time, 'available': time, 'value': value, 'unit': unit,
                'timing_basis': 'synthetic'}

    def measurements(self, stay, cutoff):
        return [self.record('index', '2020-01-03T00:30:00', 7.0)]

    def analyte_history(self, *args):
        return [self.record('prior1', '2020-01-01T01:00:00', 2.0),
                self.record('prior2', '2020-01-01T02:00:00', 3.0)]


class TinyDiagnoses:
    def prior_history(self, *args):
        return [{'token': token, 'source_version': '10', 'source_code': token,
                 'approximate_mapping': False, 'occurrences': ['old']}
                for token in ['dx:A', 'dx:B']]


KNOWLEDGE = [{'source_token': 'lab:1', 'relation': 'measures',
              'target_token': 'knowledge:A', 'source_uri': 'synthetic:reference',
              'source_version': '1', 'reviewed_by': 'fixture'}]


def history_graph(diagnoses=False, knowledge=False):
    from core.graph import build_graph
    return build_graph(HistoryStore(), {'stay_id': '1', 'subject_id': '1',
                       'cutoff': timestamp('2020-01-03T01:00:00'),
                       'sample_id': 'synthetic', 'split': 'train'},
                       KNOWLEDGE if knowledge else [], {'pain', 'fever'},
                       TinyDiagnoses() if diagnoses else None)


class GraphInformationTests(unittest.TestCase):
    def test_empty_node_view_cannot_pass_without_any_test(self):
        from core import relation_information as ri
        result = ri.audit_graph({'nodes': [], 'edges': []})
        self.assertFalse(result['fully_verified_from_raw_node_view'])

    def test_informative_flag_is_not_reconstruction_evidence(self):
        graph = tiny_graph()
        self.assertTrue(any(e['informative'] for e in graph['edges']))
        report = audit_graphs([graph])
        self.assertEqual(report['reconstruction']['graphs_fully_verified_from_raw_node_view'], 1)
        self.assertNotIn('graphs_with_extra_facts', report['reconstruction'])
        for edge in graph['edges']:
            edge['informative'] = not edge['informative']
        changed = audit_graphs([graph])
        self.assertEqual(changed['reconstruction']['graphs_fully_verified_from_raw_node_view'], 1)
        self.assertEqual(changed['annotation_mismatches'], len(graph['edges']))


    def test_same_endpoints_changed_numeric_payload_is_not_exact(self):
        graph = history_graph()
        for field in ['delta', 'interval_hours', 'rate_per_hour', 'comparable_units']:
            with self.subTest(field=field):
                altered = copy.deepcopy(graph)
                edge = next(e for e in altered['edges'] if e['relation'] == 'baseline_of')
                edge[field] = not edge[field] if field == 'comparable_units' else edge[field] + 1
                from core import relation_information as ri
                result = ri.audit_graph(altered)['relations']['baseline_of']
                self.assertTrue(result['topology']['exact'])
                self.assertFalse(result['full_exact'])
                self.assertFalse(result['payload_fields'][field]['exact'])
        self.assertTrue(ri.audit_graph(graph)['fully_verified_from_raw_node_view'])

    def test_duplicate_and_direction_are_not_discarded(self):
        from core import relation_information as ri
        graph = tiny_graph()
        edge = next(e for e in graph['edges'] if e['relation'] == 'co_complaint')
        graph['edges'].append(copy.deepcopy(edge))
        result = ri.audit_graph(graph)['relations']['co_complaint']
        self.assertFalse(result['full_exact'])
        self.assertEqual(result['topology']['unpredicted'], 1)
        graph['edges'].pop()
        edge['source'], edge['target'] = edge['target'], edge['source']
        result = ri.audit_graph(graph)['relations']['co_complaint']
        self.assertFalse(result['full_exact'])
        self.assertEqual(result['topology']['missing'], 1)
        self.assertEqual(result['topology']['unpredicted'], 1)

    def test_recurrence_recency_and_comorbidity_are_not_node_facts(self):
        from core import relation_information as ri
        graph = history_graph(diagnoses=True)
        report = ri.audit_graph(graph)
        recurrence = report['relations']['recurrence_of']
        self.assertTrue(recurrence['topology']['exact'])
        self.assertTrue(recurrence['payload_fields']['prior_encounters']['exact'])
        self.assertFalse(recurrence['payload_fields']['last_seen_hours']['tested'])
        self.assertFalse(recurrence['full_exact'])
        comorbid = report['relations']['comorbid_with']
        self.assertFalse(comorbid['topology']['tested'])
        self.assertEqual(comorbid['observed_edges'], 2)
        self.assertFalse(report['fully_verified_from_raw_node_view'])
        changed = copy.deepcopy(graph)
        edge = next(e for e in changed['edges'] if e['relation'] == 'comorbid_with')
        edge['source'], edge['target'] = edge['target'], edge['source']
        self.assertNotEqual(audit.informative_signature(graph), audit.informative_signature(changed))
        changed = copy.deepcopy(graph)
        next(e for e in changed['edges'] if e['relation'] == 'recurrence_of')['last_seen_hours'] -= 1
        self.assertNotEqual(audit.informative_signature(graph), audit.informative_signature(changed))

    def test_external_knowledge_is_disclosed_and_provenance_checked(self):
        from core import relation_information as ri
        graph = history_graph(knowledge=True)
        result = ri.audit_graph(graph)['relations']['medical:measures']
        self.assertFalse(result['topology']['tested'])
        self.assertFalse(result['full_exact'])
        report = ri.audit_graph(graph, knowledge=KNOWLEDGE)
        self.assertTrue(report['relations']['medical:measures']['full_exact'])
        self.assertFalse(report['fully_verified_from_raw_node_view'])
        edge = next(e for e in graph['edges'] if e['relation'] == 'medical:measures')
        edge['provenance']['source_version'] = 'different'
        self.assertFalse(ri.audit_graph(graph, knowledge=KNOWLEDGE)['relations']['medical:measures']['full_exact'])

    def test_unknown_fields_and_relations_are_counted_not_approved(self):
        from core import relation_information as ri
        graph = tiny_graph()
        edge = next(e for e in graph['edges'] if e['relation'] == 'co_complaint')
        edge['unseen_payload'] = 3.0
        graph['edges'].append(dict(edge, relation='unknown'))
        report = ri.audit_graph(graph)
        self.assertFalse(report['relations']['co_complaint']['full_exact'])
        self.assertEqual(report['relations']['co_complaint']['unknown_fields'], {'unseen_payload': 1})
        self.assertEqual(report['unknown_relations'], {'unknown': 1})
        self.assertFalse(report['fully_verified_from_raw_node_view'])


    def test_missing_payload_is_distinct_from_verified_null(self):
        from core import relation_information as ri
        graph = history_graph()
        edge = next(e for e in graph['edges'] if e['relation'] == 'baseline_of')
        del edge['delta']
        self.assertFalse(ri.audit_graph(graph)['relations']['baseline_of']['payload_fields']['delta']['exact'])
        graph = history_graph()
        edge = next(e for e in graph['edges'] if e['relation'] == 'baseline_of')
        next(n for n in graph['nodes'] if n['id'] == edge['source'])['unit'] = 'different'
        edge.update(delta=None, rate_per_hour=None, comparable_units=False)
        result = ri.audit_graph(graph)['relations']['baseline_of']
        self.assertTrue(result['full_exact'])

    def test_recurrence_null_still_unverified_and_summary_counts_fields(self):
        from core import relation_information as ri
        graph = history_graph(diagnoses=True)
        for edge in graph['edges']:
            if edge['relation'] == 'recurrence_of':
                edge['last_seen_hours'] = None
        summary = ri.summarize_audits([ri.audit_graph(graph)])
        self.assertEqual(summary['nonreconstructed_field_values']['recurrence_of.last_seen_hours'], 2)
        self.assertEqual(summary['relations']['recurrence_of']['fields']['last_seen_hours']['tested_graphs'], 0)

    def test_cli_reports_actual_prefix_count_and_preserves_existing_output(self):
        module = 'core.'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'graphs.jsonl').write_text(json.dumps(tiny_graph()) + '\n' + json.dumps(history_graph()) + '\n')
            for suffix in ['audit', 'relation_information']:
                result = subprocess.run([sys.executable, '-m', module + suffix,
                                         '--artifact', directory, '--limit', '1'],
                                        capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout)['graphs'], 1)
            output = root / 'existing.json'
            output.write_text('unchanged')
            result = subprocess.run([sys.executable, '-m', module + 'audit', '--artifact', directory,
                                     '--limit', '1', '--out', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), 'unchanged')


def result_row(seed, score, **changes):
    binding = {
        'artifact_graphs_sha256': 'a' * 64, 'artifact_schema': 'synthetic',
        'targets_sha256': 'b' * 64, 'source_code': {'train.py': 'c' * 64},
        'logic_contract_version': 'clinical_graph_logic_v2',
        'evaluation_version': 'visit_targets_patient_equal_v2',
        'preprocessing_schema_version': 'synthetic_features_v1',
        'class_order': ['A', 'B'], 'num_classes': 2, 'seed': seed,
        'counts': {'train': 8, 'validation': 4}, 'epochs': 3, 'batch_size': 2,
        'parameter_count': 12, 'conv': 'edge_conditioned', 'message_passing': True,
        'selection': 'best validation macro_f1', 'test_evaluated': False,
    }
    binding.update(changes)
    return {'binding': binding, 'metrics': {'macro_f1': score},
            'relation_separation': {'at_init': [], 'at_end': []}}


class AggregationTests(unittest.TestCase):
    def test_active_capacity_is_disclosed_without_blocking_nomp_comparison(self):
        from core import aggregate
        rows = {'nomp_seed1': result_row(1, .2, message_passing=False, active_parameter_count=4),
                'main_seed1': result_row(1, .3, active_parameter_count=12)}
        report = aggregate.aggregate_rows(rows)
        self.assertEqual(report['n_cohorts'], 1)
        comparison = report['cohorts'][0]['comparisons'][0]
        self.assertEqual(comparison['treatment_differences']['active_parameter_count'],
                         {'reference': 4, 'arm': 12})
        self.assertIn('active', aggregate.render_report(report))

    def test_rewiring_realization_counts_are_outcomes_not_cohort_or_treatment_policy(self):
        from core import aggregate
        rows = {f'rewired_seed{seed}': result_row(
            seed, .2, rewired_relations=['co_complaint'], rewiring_policy='bounded_null_v1',
            rewire_seed=seed, rewiring={'train': {'changed_edges': seed}})
            for seed in (1, 2)}
        report = aggregate.aggregate_rows(rows)
        self.assertEqual(report['n_cohorts'], 1)
        self.assertEqual(report['cohorts'][0]['arms']['rewired']['n_seeds'], 2)
        self.assertEqual(report['runs']['rewired_seed1']['binding']['rewiring'],
                         {'train': {'changed_edges': 1}})

    def test_rewired_comparison_requires_payload_disabled_in_both_arms(self):
        from core import aggregate
        rows = {
            'nomp_seed1': result_row(1, .2, message_passing=False, edge_payload=True),
            'plain_seed1': result_row(1, .25, edge_payload=True),
            'rewired_seed1': result_row(1, .3, edge_payload=False,
                                        rewired_relations=['co_complaint'],
                                        rewiring_policy='bounded_null_v1', rewire_seed=1),
        }
        report = aggregate.aggregate_rows(rows)
        cohort = report['cohorts'][0]
        self.assertFalse(cohort['comparison_eligible'])
        self.assertFalse(cohort['comparisons'])
        self.assertEqual(cohort['ineligible_comparisons'][0]['reason'],
                         'rewiring requires edge_payload=false in both arms')
        rows['nomp_seed1']['binding']['edge_payload'] = False
        matched = aggregate.aggregate_rows(rows)['cohorts'][0]
        self.assertTrue(matched['comparison_eligible'])
        self.assertEqual(matched['comparisons'][0]['status'],
                         'binding_matched_descriptive_only')

    def test_contradictory_class_universe_is_rejected(self):
        from core import aggregate
        with self.assertRaisesRegex(ValueError, 'class'):
            aggregate.aggregate_rows({'main_seed1': result_row(1, .2, num_classes=3)})

    def test_cli_handles_hgt_and_standalone_without_choosing_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            row = result_row(1, .2, conv='hgt')
            row['relation_separation'] = {'at_init': [{'k_rel': .3}], 'at_end': [{'k_rel': .1}]}
            for name, value in [('typed_seed1', row),
                                ('tabular', {'method': 'xgboost', 'metrics': {'macro_f1': .9}, 'features': 7})]:
                root = Path(directory) / name
                root.mkdir()
                (root / 'result.json').write_text(json.dumps(value))
            command = [sys.executable, '-m', 'core.aggregate', directory]
            result = subprocess.run(command + ['--json'], capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report['n_runs'], 2)
            self.assertEqual(report['n_cohorts'], 2)
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn('BG-HGNN', result.stdout)
            self.assertIn('tabular', result.stdout)
            self.assertNotIn('EN IYI ARM', result.stdout)

    def test_filename_seed_is_not_a_saved_seed_binding(self):
        from core import aggregate
        row = result_row(1, .2)
        del row['binding']['seed']
        report = aggregate.aggregate_rows({'main_seed1': row, 'nomp_seed1': copy.deepcopy(row)})
        self.assertEqual(report['cohorts'][0]['comparisons'][0]['paired_seeds'], [])

    def test_matched_seed_deltas_not_global_noise_threshold(self):
        from core import aggregate
        rows = {'nomp_seed1': result_row(1, .2, message_passing=False),
                'nomp_seed2': result_row(2, .2, message_passing=False),
                'main_seed1': result_row(1, .3), 'main_seed2': result_row(2, .5),
                'main_seed3': result_row(3, .9)}
        report = aggregate.aggregate_rows(rows)
        cohort = report['cohorts'][0]
        self.assertEqual(cohort['arms']['main']['seeds'], ['1', '2', '3'])
        self.assertEqual(cohort['arms']['nomp']['n_runs'], 2)
        comparison = cohort['comparisons'][0]
        self.assertEqual(comparison['paired_seeds'], ['1', '2'])
        self.assertEqual(comparison['unpaired_arm_seeds'], ['3'])
        self.assertAlmostEqual(comparison['paired_metrics']['macro_f1']['mean'], .2)
        self.assertIsNone(comparison['descriptive_delta_over_reference_sd'])
        self.assertEqual(report['patient_cluster_uncertainty']['status'], 'not_measured')
        text = aggregate.render_report(report)
        self.assertNotIn('EN IYI ARM', text)
        self.assertNotIn('GURULTU', text)
        self.assertNotIn('winner', report)

    def test_incompatible_bindings_are_partitioned_even_with_same_arm_name(self):
        from core import aggregate
        cases = {'artifact_graphs_sha256': 'd' * 64, 'class_order': ['B', 'A'],
                 'preprocessing_schema_version': 'different', 'epochs': 4,
                 'source_code': {'train.py': 'e' * 64},
                 'evaluation_version': 'different', 'logic_contract_version': 'different',
                 'parameter_count': 24, 'feature_contract_sha256': 'f' * 64,
                 'counts': {'train': 8, 'validation': 3}}
        for field, value in cases.items():
            with self.subTest(field=field):
                report = aggregate.aggregate_rows({'nomp_seed1': result_row(1, .2),
                                                   'nomp_seed2': result_row(2, .3, **{field: value})})
                self.assertEqual(len(report['cohorts']), 2)
                self.assertTrue(all(not c['comparisons'] for c in report['cohorts']))

    def test_missing_metadata_is_not_assumed_matched(self):
        from core import aggregate
        old = result_row(1, .2)
        for key in ['evaluation_version', 'logic_contract_version', 'preprocessing_schema_version']:
            del old['binding'][key]
        rows = {'nomp_seed1': old, 'main_seed1': copy.deepcopy(old)}
        report = aggregate.aggregate_rows(rows)
        cohort = report['cohorts'][0]
        self.assertEqual(cohort['contract']['evaluation_version'], 'legacy_unspecified')
        self.assertFalse(cohort['comparison_eligible'])
        self.assertFalse(cohort['comparisons'])
        self.assertIn('preprocessing_schema_version_or_feature_hash', cohort['missing_bindings'])

    def test_single_seed_and_standalone_are_not_fake_replicates(self):
        from core import aggregate
        row = result_row(7, .6, conv='hgt')
        report = aggregate.aggregate_rows({'typed': row})
        stats = report['cohorts'][0]['arms']['typed']
        self.assertEqual(stats['seeds'], ['7'])
        self.assertIsNone(stats['metrics']['macro_f1']['sample_sd'])
        self.assertIn('relation_separation', report['runs']['typed'])
        with self.assertRaisesRegex(ValueError, 'seed'):
            aggregate.aggregate_rows({'main_seed8': row})


if __name__ == '__main__':
    unittest.main()
