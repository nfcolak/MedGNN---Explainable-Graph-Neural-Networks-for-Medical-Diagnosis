"""Corrected clinical runs must not silently accept historical artifact contracts."""
import pytest
from comparison.standardized.clinical_graph_v2 import train


def test_historical_artifact_is_rejected_before_a_corrected_run():
    assert hasattr(train, 'validate_artifact_manifest'), 'training lacks input version guard'
    with pytest.raises(ValueError, match='rebuild'):
        train.validate_artifact_manifest({'status': 'completed', 'temporal_clean': True})


def test_rewiring_with_numeric_payload_is_refused():
    with pytest.raises(ValueError, match='no-edge-payload'):
        train.validate_control_configuration(['baseline_of'], True)
    train.validate_control_configuration(['baseline_of'], False)
    train.validate_control_configuration([], True)


def test_corrected_manifest_requires_explicit_temporal_uncertainty():
    manifest = {'status': 'completed', 'logic_contract_version': 'clinical_graph_logic_v2',
                'temporal_clean': False,
                'visit_membership_contract_version': 'clinical_visit_membership_v1',
                'visit_membership_file': 'visit_membership.jsonl',
                'visit_membership_rows': 0, 'visit_membership_sha256': '0' * 64}
    train.validate_artifact_manifest(manifest)
    manifest['temporal_clean'] = True
    with pytest.raises(ValueError, match='temporal_clean'):
        train.validate_artifact_manifest(manifest)


def test_target_loader_rejects_duplicate_visits_and_cross_fold_patients(tmp_path):
    target = tmp_path / 'targets.csv'
    header = 'sample_id,target,split,subject_id\n'
    target.write_text(header + 'a,0,train,p\na,0,train,p\n')
    with pytest.raises(ValueError, match='Duplicate'):
        train.load_targets(target)
    target.write_text(header + 'a,0,train,p\nb,0,validation,p\n')
    with pytest.raises(ValueError, match='fold'):
        train.load_targets(target)


def test_target_binding_requires_same_cohort_and_label_order(tmp_path):
    import json
    from comparison.standardized.clinical_graph_v2 import contracts
    from comparison.standardized.clinical_graph_v2.schema import sha256
    assert hasattr(contracts, 'verify_target_binding'), 'target lineage is not checked'
    target = tmp_path / 'targets.csv'
    target.write_text('sample_id,target,split,subject_id\na,0,train,p\n')
    binding = {'status': 'completed', 'cohort_sha256': 'cohort-a',
               'labels': ['a', 'b'], 'artifact_files': {'targets.csv': sha256(target)}}
    path = tmp_path / 'binding_manifest.json'
    path.write_text(json.dumps(binding))
    manifest = {'inherited_cohort_sha256': 'cohort-a'}
    contracts.verify_target_binding(manifest, target, ['a', 'b'])
    with pytest.raises(ValueError, match='label order'):
        contracts.verify_target_binding(manifest, target, ['b', 'a'])
    with pytest.raises(ValueError, match='cohort'):
        contracts.verify_target_binding({'inherited_cohort_sha256': 'cohort-b'},
                                         target, ['a', 'b'])
