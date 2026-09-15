"""Train-only transformations of explicitly audited source-snapshot fields."""
import numpy as np
import pandas as pd
import hashlib

from .spec import ALLOWED


def validate_spec(spec):
    names = [f['source'] for f in spec]
    if len(names) != len(set(names)) or any(ALLOWED.get(f['source']) != f['kind'] for f in spec):
        raise ValueError('Feature is outside the audited allowlist')


def align_source(frame, split, arrays):
    """Use original CSV order (legacy builder order), never sort/reindex by label."""
    keys = frame.subject_id.astype(str)
    if keys.duplicated().any():
        raise ValueError('Duplicate source patients')
    keep = keys.isin(split['fold'])
    aligned = frame.loc[keep].copy().reset_index(drop=True)
    keys = aligned.subject_id.astype(str)
    if set(keys) != set(split['fold']):
        raise ValueError('Canonical patient coverage mismatch')
    labels = aligned.disease_1.map({c: i for i, c in enumerate(split['classes'])}).to_numpy()
    folds = keys.map(split['fold']).to_numpy()
    if not np.array_equal(labels, arrays['y']) or not np.array_equal(folds, arrays['folds']):
        raise ValueError('Canonical row order, labels or folds mismatch')
    return aligned, hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def fit_transform(frame, folds, spec):
    """Retain every row; fit observed mean/population scale on fold zero only."""
    validate_spec(spec)
    train = np.asarray(folds) == 0
    if train.shape != (len(frame),) or not train.any():
        raise ValueError('Aligned, nonempty training fold required')
    columns, names, statistics, coverage = [], [], {}, {}
    for field in spec:
        name = field['source']
        value = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float)
        value[~np.isfinite(value)] = np.nan
        if 'range' in field:
            low, high = field['range']
            value[(value < low) | (value > high)] = np.nan
        observed = np.isfinite(value)
        fit = value[train & observed]
        mean = float(fit.mean()) if len(fit) else 0.
        scale = float(fit.std()) if len(fit) else 1.
        if scale == 0:
            scale = 1.
        if field['kind'] == 'binary':
            value[~np.isin(value, [0., 1.])] = np.nan
            observed = np.isfinite(value)
            mean, scale = 0., 1.
        encoded = np.where(observed, (value - mean) / scale, 0.)
        columns.extend([encoded, (~observed).astype(float)])
        names.extend([name, name + '__missing'])
        statistics[name] = {'mean': mean, 'scale': scale,
                            'observed_train': int((train & observed).sum()),
                            'imputation': 'zero for binary; observed train mean for numeric'}
        coverage[name] = {'observed': int(observed.sum()), 'missing': int((~observed).sum()),
                          'observed_by_fold': [int((observed & (folds == f)).sum()) for f in range(3)],
                          'positive': int((value == 1).sum()) if field['kind'] == 'binary' else None}
    # Stable documented source sex categories; conflicting/missing/unrecognized => unknown.
    female = frame['gender_F'].map({True: 1., False: 0., 'True': 1., 'False': 0., '1': 1., '0': 0.}).to_numpy()
    male = frame['gender_M'].map({True: 1., False: 0., 'True': 1., 'False': 0., '1': 1., '0': 0.}).to_numpy()
    f, m = (female == 1) & (male == 0), (male == 1) & (female == 0)
    columns.extend([f.astype(float), m.astype(float), (~(f | m)).astype(float)])
    names.extend(['sex_F', 'sex_M', 'sex_unknown'])
    coverage['sex'] = {'female': int(f.sum()), 'male': int(m.sum()), 'unknown': int((~(f | m)).sum())}
    result = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transformed feature')
    return result, {'names': names, 'statistics': statistics,
                    'fit_scope': 'canonical train only', 'sex_categories': ['F', 'M', 'unknown']}, coverage
