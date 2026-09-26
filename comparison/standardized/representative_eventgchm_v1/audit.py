"""Read-only input profiling and stored-validation reporting for one approved run."""
from collections import Counter
import csv
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_recall_fscore_support


def quotas(counts):
    counts = np.asarray(counts, dtype=np.int64)
    allocation = np.full(30, 60, dtype=np.int64)
    assert (counts >= allocation).all()
    remaining = 6000 - int(allocation.sum())
    # Original class frequencies, not capacity frequencies. Redistribute only if capped.
    while remaining:
        eligible = allocation < counts
        raw = remaining * counts * eligible / (counts * eligible).sum()
        extra = np.minimum(np.floor(raw).astype(int), counts - allocation)
        allocation += extra
        remaining = 6000 - int(allocation.sum())
        if remaining:
            order = sorted(np.flatnonzero(allocation < counts), key=lambda c: (-(raw[c] - np.floor(raw[c])), int(c)))
            for c in order[:remaining]:
                allocation[c] += 1
            remaining = 6000 - int(allocation.sum())
    assert allocation.sum() == 6000 and allocation.min() >= 60
    return allocation


def choose(rows, target_rows):
    rng = np.random.default_rng(1234)
    counts = np.bincount([r[3] for r in rows], minlength=30)
    allocation = quotas(counts)
    groups = {c: {} for c in range(30)}
    for row in rows:
        groups[row[3]].setdefault(target_rows[row[0]]['subject_id'], []).append(row)
    selected, used_subjects = [], set()
    # Scarce classes first; unseen subjects first, then already used subjects.
    # One seeded random visit per subject; no graph-size or outcome selection.
    for c in sorted(range(30), key=lambda c: (len(groups[c]) / allocation[c], c)):
        subjects = sorted(groups[c])
        subjects = [subjects[int(i)] for i in rng.permutation(len(subjects))]
        subjects.sort(key=lambda s: s in used_subjects)
        need = int(allocation[c])
        first = subjects[:need]
        for subject in first:
            visits = groups[c][subject]
            selected.append(visits[int(rng.integers(len(visits)))])
            used_subjects.add(subject)
        if len(first) < need:
            taken = {r[0] for r in selected}
            extras = [r for visits in groups[c].values() for r in visits if r[0] not in taken]
            selected.extend(extras[int(i)] for i in rng.permutation(len(extras))[:need-len(first)])
    selected = [selected[int(i)] for i in rng.permutation(len(selected))]
    assert len(selected) == len({r[0] for r in selected}) == 6000
    assert np.array_equal(np.bincount([r[3] for r in selected], minlength=30), allocation)
    # This cohort can achieve the global maximum: one patient per selected visit.
    assert len({target_rows[r[0]]['subject_id'] for r in selected}) == 6000, 'Diversity requires explicit matching, not a silent approximation'
    return selected, counts.tolist(), allocation.tolist()


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(('min','p25','median','p75','p90','p95','p99','max'), np.quantile(a, [0,.25,.5,.75,.9,.95,.99,1]).tolist()))


def metadata_profile(rows, targets):
    counts = Counter(targets[r[0]]['subject_id'] for r in rows)
    return {'visits': len(rows), 'patients': len(counts), 'multivisit_patients': sum(v > 1 for v in counts.values()),
            'multivisit_patient_proportion': sum(v > 1 for v in counts.values()) / len(counts),
            'visits_from_multivisit_patients_proportion': sum(v for v in counts.values() if v > 1) / len(rows),
            'class_counts': np.bincount([r[3] for r in rows], minlength=30).tolist(),
            'nodes': quantiles([r[4] for r in rows]), 'edges': quantiles([r[5] for r in rows])}


def graph_profile(graph):
    events = [n for n in graph['nodes'] if n['kind'] == 'event']
    index = {n['id'] for n in graph['nodes'] if n['kind'] == 'visit' and n['token'] == 'visit:index'}
    current = {e['target'] for e in graph['edges'] if e['relation'] == 'contains_event' and e['source'] in index}
    return {'events': len(events), 'current_events': sum(n['id'] in current for n in events),
            'history_events': sum(n['id'] not in current for n in events),
            'prior_visits': graph['coverage']['prior_visits'], 'knowledge_edges': graph['coverage']['knowledge_edges'],
            'event_value_missing': sum(n.get('value') is None for n in events),
            'event_time_missing': sum(n.get('time_hours') is None for n in events),
            'event_availability_missing': sum(n.get('available_hours') is None for n in events),
            'event_unit_missing': sum(not n.get('unit') for n in events)}


def summarize_profiles(rows):
    return {'graphs': len(rows), 'quantiles': {k: quantiles([r[k] for r in rows]) for k in rows[0]},
            'means': {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
            'any_history_fraction': float(np.mean([r['prior_visits'] > 0 for r in rows])),
            'any_knowledge_fraction': float(np.mean([r['knowledge_edges'] > 0 for r in rows])),
            'event_empty_fraction': float(np.mean([r['events'] == 0 for r in rows]))}


def report(root, state, native):
    data = np.load(root / 'validation.npz')
    y, proba = data['y'], data['proba']
    metadata = json.loads((root / 'selected_samples.json').read_text())
    val = [r for r in metadata if r['split'] == 'validation']
    assert len(val) == len(y) == 9582 and proba.shape == (9582,30)
    assert np.isfinite(proba).all() and np.allclose(proba.sum(1), 1, atol=1e-6)
    assert np.array_equal(y, [r['target'] for r in val])
    assert np.array_equal(data['sample_id'], [r['sample_id'] for r in val])
    subjects, patient = np.unique([r['subject_id'] for r in val], return_inverse=True)
    counts = np.bincount(patient)
    assert len(subjects) == 5620
    weights = 1.0 / counts[patient]
    pred = proba.argmax(1)
    def metrics(w=None):
        return {'accuracy': float(accuracy_score(y, pred, sample_weight=w)),
                'macro_f1': float(f1_score(y, pred, labels=np.arange(30), average='macro', zero_division=0, sample_weight=w)),
                'balanced_accuracy': float(recall_score(y, pred, labels=np.arange(30), average='macro', zero_division=0, sample_weight=w))}
    visit, patient_equal = metrics(), metrics(weights)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(30), zero_division=0)
    correct = np.bincount(patient, weights=(pred == y).astype(float))
    rng = np.random.default_rng(91234)
    bootstrap_visit, bootstrap_patient = [], []
    for _ in range(500):
        draw = rng.integers(len(subjects), size=len(subjects))
        bootstrap_visit.append(float(correct[draw].sum()/counts[draw].sum()))
        bootstrap_patient.append(float((correct[draw]/counts[draw]).mean()))
    train_counts = json.loads((root/'sample_coverage.json').read_text())['train']['available_class_counts']
    majority = int(np.argmax(train_counts))
    result = {'status': 'completed', 'selected_epoch': state['best_epoch'], 'visit_metrics': visit,
              'patient_equal_metrics': patient_equal, 'visits': len(y), 'patients': len(subjects),
              'patient_cluster_bootstrap': {'resamples': 500, 'seed': 91234, 'confidence': .95,
                 'visit_accuracy_interval': np.quantile(bootstrap_visit,[.025,.975]).tolist(),
                 'patient_equal_accuracy_interval': np.quantile(bootstrap_patient,[.025,.975]).tolist(),
                 'scope': 'Fixed validation-selected checkpoint; sample patients with replacement retaining all their visits; percentile intervals; excludes seed/model-selection uncertainty; no macro-F1 bootstrap'},
              'train_majority_baseline': {'class_index': majority, 'class': state['class_order'][majority], 'natural_validation_accuracy': float(np.mean(y == majority)), 'chosen_on': 'full_train_labels_only'},
              'per_class': [{'index': c, 'label': state['class_order'][c], 'precision': float(precision[c]), 'recall': float(recall[c]), 'f1': float(f1[c]), 'support': int(support[c])} for c in range(30)],
              'test_evaluated': False, 'temporal_clean': False,
              'caveats': ['Single seed, validation reused for epoch selection, not final held-out test.', 'Deliberately patient-diverse, minimum-quota training is not a natural visit sample.', 'History/current-event/missingness comparison uses a separate seeded 1000-graph reference, not all 77930 payloads.', 'storetime is a recorded-availability proxy; knowledge relations not clinically reviewed.', 'Different inputs/cohorts: no superiority claim against prior methods.']}
    assert abs(visit['macro_f1'] - state['best_validation_macro_f1']) < 1e-12
    native.atomic_json(root/'metrics.json', result)
    with (root/'per_class.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['per_class'][0])); writer.writeheader(); writer.writerows(result['per_class'])
    ci = result['patient_cluster_bootstrap']['visit_accuracy_interval']
    text = f"# Representative local EventGCHM\n\n6000 training visits / 6000 patients; 9582 natural validation visits / 5620 patients. Seed 1234; 10 epochs; selected epoch {state['best_epoch']} by all-30-class validation macro-F1.\n\nVisit accuracy: {visit['accuracy']:.6f}; macro-F1: {visit['macro_f1']:.6f}; balanced accuracy: {visit['balanced_accuracy']:.6f}.\n\nPatient-equal accuracy: {patient_equal['accuracy']:.6f}; macro-F1: {patient_equal['macro_f1']:.6f}. Patient-cluster 95% accuracy interval: [{ci[0]:.6f}, {ci[1]:.6f}] (500 bootstrap samples).\n\nNatural validation train-majority baseline accuracy: {result['train_majority_baseline']['natural_validation_accuracy']:.6f}.\n\nAdamW lr=0.001 weight_decay=1e-5; inverse-square-root selected-training class weights; weighted accumulation denominator; microbatch 1, accumulation 8. CPU/BLAS/Torch/inter-op threads 1; niceness 19.\n\nPreprocessing, vocabulary, scaler and degree histogram fit on the selected 6000 only. No graph-size filtering or truncation. Exact full-training metadata and sampled payload comparisons: distribution_profile.json. This intentionally eliminates repeated patients in selected training; it does not preserve the source repeat-visit distribution. Class proportions also change by policy.\n\n" + '\n'.join('- '+x for x in result['caveats']) + '\n'
    (root/'report.md').write_text(text)
    return result
