"""Opt-in enrichment: synthetic-only fixtures; never print clinical records."""
import importlib

import numpy as np
import pandas as pd
import pytest


def enriched():
    try:
        return importlib.import_module('comparison.standardized.enriched_input_v1.features')
    except ModuleNotFoundError:
        pytest.fail('Versioned enriched feature pipeline is not implemented')


def test_train_only_numeric_sex_and_missingness():
    module = enriched()
    df = pd.DataFrame({'age': [20., 40., 9000., np.nan],
                       'gender_F': [True, False, False, True],
                       'gender_M': [False, True, False, True]})
    spec = [{'source': 'age', 'kind': 'numeric', 'unit': 'anchor years',
             'availability': 'anchor_demographic'}]
    x, recipe, coverage = module.fit_transform(df, np.array([0, 0, 1, 2]), spec)
    assert recipe['statistics']['age']['mean'] == 30.
    assert recipe['statistics']['age']['scale'] == 10.
    assert recipe['names'] == ['age', 'age__missing', 'sex_F', 'sex_M', 'sex_unknown']
    np.testing.assert_array_equal(x[:, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(x[:, -1], [0, 0, 1, 1])
    assert x[2, 0] == 897. and x[3, 0] == 0.
    assert coverage['age']['observed'] == 3 and np.isfinite(x).all()
    changed = df.copy(); changed.loc[2, 'age'] = -1e8
    assert module.fit_transform(changed, np.array([0, 0, 1, 2]), spec)[1] == recipe


def test_audited_allowlist_rejects_target_and_unrequested_fields():
    module = enriched()
    assert hasattr(module, 'validate_spec'), 'Audited allowlist validation missing'
    for bad in ['disease_1', 'icd_codes', 'disposition', 'los_hours', 'n_ed_visits',
                'vs_sbp_mean', 'race_WHITE', 'bmi', 'hx_disease_1', 'lab_target_proxy']:
        with pytest.raises(ValueError, match='allowlist'):
            module.validate_spec([{'source': bad, 'kind': 'numeric'}])
    module.validate_spec([{'source': 'age', 'kind': 'numeric'}])


def test_mapping_fails_closed_on_patient_order_label_and_fold():
    module = enriched()
    assert hasattr(module, 'align_source'), 'Canonical alignment missing'
    frame = pd.DataFrame({'subject_id': [11, 22, 33], 'disease_1': ['A', 'B', 'A']})
    split = {'fold': {'11': 0, '22': 1, '33': 2}, 'classes': ['A', 'B']}
    arrays = {'y': np.array([0, 1, 0]), 'folds': np.array([0, 1, 2])}
    aligned, digest = module.align_source(frame, split, arrays)
    assert len(aligned) == 3 and len(digest) == 64
    for altered in [frame.iloc[::-1], frame.iloc[:2], pd.concat([frame, frame.iloc[:1]])]:
        with pytest.raises(ValueError):
            module.align_source(altered, split, arrays)
    for key in ['y', 'folds']:
        with pytest.raises(ValueError):
            module.align_source(frame, split, {**arrays, key: arrays[key][::-1] + 1})


def test_history_excludes_current_equal_start_and_unfinished_visits():
    try:
        audit = importlib.import_module('comparison.standardized.enriched_input_v1.audit')
    except ModuleNotFoundError:
        pytest.fail('History lineage guard missing')
    frame = pd.DataFrame({'subject_id': [11, 22], 'hx_anemia': [1., 1.]})
    index = pd.DataFrame({'subject_id': [11, 22], 'stay_id': [102, 202],
                          'intime': ['2020-01-02', '2020-01-02']})
    stays = pd.DataFrame({'subject_id': [11, 11, 11, 22, 22],
                          'stay_id': [100, 101, 102, 201, 202],
                          'intime': ['2020-01-01', '2020-01-02', '2020-01-02', '2020-01-01', '2020-01-02'],
                          'outtime': ['2020-01-01 12:00', '2020-01-02 12:00', '2020-01-03', '2020-01-03', '2020-01-03']})
    diagnoses = pd.DataFrame({'subject_id': [11, 11, 22, 22], 'stay_id': [100, 102, 201, 202],
                              'history_field': ['hx_anemia'] * 4})
    clean, evidence = audit.guard_history(frame, index, stays, diagnoses, ['hx_anemia'])
    assert clean.hx_anemia.iloc[0] == 1
    assert clean.hx_anemia.iloc[1] == 0  # no prior-ended evidence, not disease absence
    assert evidence['unsupported_positive_values_removed'] == 1
    assert evidence['documentation_time_verified'] is False
    flipped = frame.copy(); flipped['hx_anemia'] = 0.
    clean_flipped, _ = audit.guard_history(flipped, index, stays, diagnoses, ['hx_anemia'])
    np.testing.assert_array_equal(clean.hx_anemia, clean_flipped.hx_anemia,
                                  err_msg='Unsafe source history must not survive through missingness')
    # A current-only or equal-start diagnosis must never count as prior history.
    clean, _ = audit.guard_history(frame, index, stays,
                                   diagnoses[diagnoses.stay_id != 100], ['hx_anemia'])
    assert clean.hx_anemia.eq(0).all()


def test_lineage_recovery_does_not_choose_arbitrary_duplicate_stay(tmp_path):
    from comparison.standardized.enriched_input_v1 import audit
    assert hasattr(audit, 'audit_source'), 'Raw lineage reconstruction missing'
    frame = pd.DataFrame({'subject_id': [11, 22], 'temperature': [37., 37.],
                          'heartrate': [80., 80.], 'resprate': [16., 16.],
                          'o2sat': [99., 99.], 'sbp': [120., 120.], 'dbp': [80., 80.],
                          'acuity': [2, 2], 'hx_anemia': [1., 1.]})
    triage = pd.concat([frame.iloc[[0]], frame.iloc[[1]], frame.iloc[[1]]]).copy()
    triage['stay_id'] = [100, 200, 201]; triage['temperature'] = 98.6
    triage.drop(columns='hx_anemia').to_csv(tmp_path / 'triage.csv', index=False)
    pd.DataFrame({'subject_id': [11, 22, 22], 'stay_id': [100, 200, 201],
                  'intime': ['2020-01-01'] * 3, 'outtime': ['2020-01-02'] * 3}).to_csv(tmp_path / 'edstays.csv', index=False)
    pd.DataFrame({'subject_id': [11, 22], 'stay_id': [100, 200], 'icd_code': ['D649', 'D649'],
                  'icd_version': [10, 10], 'icd_title': ['Anemia, unspecified'] * 2}).to_csv(tmp_path / 'diagnosis.csv', index=False)
    pd.DataFrame({'icd9_code': ['1'], 'icd10_code': ['D649'], 'no_map': [0], 'approximate': [0]}).to_csv(tmp_path / 'icd9_to_icd10_mapping.csv', index=False)
    clean, evidence = audit.audit_source(frame, tmp_path, ['hx_anemia'])
    assert evidence['lineage']['unique_matches'] == 1
    assert evidence['lineage']['ambiguous_or_unmatched'] == 1
    assert clean.hx_anemia.iloc[0] == 0  # only CURRENT diagnoses exist
    assert np.isnan(clean.hx_anemia.iloc[1])  # index remains ambiguous


def legacy_fixture(tmp_path):
    import json
    from shared.lib.benchmark_contract import file_sha256
    root = tmp_path / 'legacy'; root.mkdir()
    frame = pd.DataFrame({'subject_id': [11, 22, 33], 'disease_1': ['A', 'B', 'A'],
                          'age': [20., 40., np.nan], 'gender_F': [1, 0, 0], 'gender_M': [0, 1, 0],
                          'med_a': [1, 0, 1], 'chiefcomplaint_1': ['', 'x', '']})
    source = tmp_path / 'source.csv'; frame.to_csv(source, index=False)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'fold': {'11': 0, '22': 1, '33': 2}, 'classes': list('ABCDEF')}))
    np.savez_compressed(root / 'inputs_no_identifiers.npz', presence=np.array([[1, 0], [0, 1], [1, 0]]),
                        y=np.array([0, 1, 0]), folds=np.array([0, 1, 2]))
    hashes = {'input_sha256': file_sha256(root / 'inputs_no_identifiers.npz'),
              'source_sha256': file_sha256(source), 'split_sha256': file_sha256(split)}
    contract = dict(hashes, names=['med:a', 'cc:x'], classes=list('ABCDEF'), concept_channels=2,
                    effective_feature_dim=3, patient_specific_hub_channels=0, temporal_clean=False,
                    raw_to_model_train_only=False,
                    fit_scope='canonical train rows of already-filtered source snapshot only',
                    counts=[1, 1, 1], fit_rows=1, audited_graphs=3)
    (root / 'input_contract_and_audit.json').write_text(json.dumps(contract))
    trial = root / 'trials/method/variant'; trial.mkdir(parents=True)
    (trial / 'run_manifest.json').write_text(json.dumps(dict(status='completed', topology='star',
                                                           n_train=1, n_validation=1, binding=hashes)))
    return root, source, split


def test_versioned_build_dryrun_no_overwrite_and_exact_npz(tmp_path):
    try:
        build = importlib.import_module('comparison.standardized.enriched_input_v1.build')
    except ModuleNotFoundError:
        pytest.fail('Versioned builder missing')
    root, source, split = legacy_fixture(tmp_path)
    output = tmp_path / 'enriched'
    plan = build.build_input(root, source, split, tmp_path / 'no_raw', output, execute=False)
    assert plan['status'] == 'dry_run' and not output.exists()
    manifest = build.build_input(root, source, split, tmp_path / 'no_raw', output, execute=True)
    assert manifest['status'] == 'completed'
    bundle = build.load_enriched(output)
    with np.load(root / 'inputs_no_identifiers.npz') as old:
        for key in ['presence', 'y', 'folds']:
            np.testing.assert_array_equal(bundle[key], old[key])
    assert bundle['hub_numeric'].shape[0] == 3 and np.isfinite(bundle['hub_numeric']).all()
    assert manifest['temporal_clean'] is False and manifest['raw_to_model_train_only'] is False
    with pytest.raises(FileExistsError):
        build.build_input(root, source, split, tmp_path / 'no_raw', output, execute=True)
    # Tampering must fail before the model can consume changed preprocessing.
    import json
    metadata = json.loads((output / 'manifest.json').read_text())
    metadata['recipe']['names'][0] = 'disease_1'
    (output / 'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        build.load_enriched(output)


def test_bounded_real_train_checkpoint_replay_and_dryrun(tmp_path):
    try:
        run = importlib.import_module('comparison.standardized.enriched_input_v1.run')
    except ModuleNotFoundError:
        pytest.fail('Enriched train/checkpoint runner missing')
    from comparison.standardized.enriched_input_v1.build import build_input, load_enriched
    root, source, split = legacy_fixture(tmp_path)
    artifact = tmp_path / 'input'; output = tmp_path / 'run'
    build_input(root, source, split, tmp_path / 'no_raw', artifact, execute=True)
    config = {'seed': 1, 'width': 8, 'layers': 1, 'rank': 2, 'interactions': True,
              'steps': 2, 'train_limit': 1, 'val_limit': 1, 'batch_size': 1,
              'lr': .001, 'weight_decay': 1e-5, 'threads': 1,
              'cross_pairs': True, 'no_messages': False}
    assert run.main(['--input-root', str(artifact), '--output-dir', str(output), '--dry-run']) == 0
    assert not output.exists()
    result = run.train_smoke(artifact, output, config)
    assert result['status'] == 'completed' and result['optimizer_steps'] == 2
    replay = run.replay(output)
    assert replay['arrays_equal'] and replay['metrics_equal']
    assert replay['logits_max_abs_diff'] == 0.
    assert result['numeric_gradient_l1'] > 0 and result['numeric_logit_max_abs_delta'] > 0
    assert result['test_evaluated'] is False
    with pytest.raises(FileExistsError):
        run.train_smoke(artifact, output, config)
