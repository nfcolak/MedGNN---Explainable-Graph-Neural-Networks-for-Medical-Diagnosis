import random
from collections import Counter


def select_top_labels(targets, k):
    """Keep the k most frequent TRAIN labels; return (kept ids, remap, dropped counts).

    Frequency is counted on the TRAIN fold only. Ranking classes by their overall
    frequency would let the validation fold decide which classes exist, which is a
    label-distribution leak even though no individual validation row is read.

    Rows outside the kept set are DROPPED, not merged into an "other" class. Merging
    would invent a heterogeneous class that no clinician would recognise and would
    quietly re-inflate the difficulty this option is removing. Ties at the boundary
    are broken by class index so the selection is deterministic.

    **This changes the task, not the model.** A k-class score is not comparable with
    a 30-class score: fewer classes means a higher random floor, a higher majority
    baseline, and the hardest rare classes removed from the denominator of macro-F1.
    Every run records `num_classes` and `dropped_rows` so the two can never be put
    in one table by accident.
    """
    counts = Counter(t for t, split, _ in targets.values() if split == 'train')
    if not counts:
        raise ValueError('No training rows to rank labels by')
    if k < 1 or k > len(counts):
        raise ValueError(f'--top-k-labels {k} exceeds the {len(counts)} training classes')
    ranked = sorted(counts, key=lambda c: (-counts[c], c))[:k]
    kept = sorted(ranked)
    remap = {c: i for i, c in enumerate(kept)}
    dropped = Counter()
    filtered = {}
    for sid, (t, split, subject) in targets.items():
        if t in remap:
            filtered[sid] = (remap[t], split, subject)
        else:
            dropped[split] += 1
    return filtered, kept, dict(dropped)


def sample_train_ids(targets, limit, sample_seed):
    """The seeded TRAIN subsample shared by every arm."""
    train_ids = {sid for sid, entry in targets.items() if entry[1] == 'train'}
    if limit is None:
        return train_ids
    rng = random.Random(sample_seed)
    return set(rng.sample(sorted(train_ids), min(limit, len(train_ids))))


def select_dev_ids(targets, train_ids, dev_limit, sample_seed, eligible=None):
    """Patient-disjoint development rows from TRAIN-fold patients outside the sample.

    Epoch and hyper-parameter selection happen on these rows, so the validation fold
    is read once per final run and the test fold stays closed. Excluding every patient
    of the drawn sample stops one patient from informing both fitting and selection.
    """
    if dev_limit is None:
        return frozenset()
    if isinstance(dev_limit, bool) or int(dev_limit) < 1:
        raise ValueError('dev_limit must be a positive integer')
    sampled_subjects = {targets[sid][2] for sid in train_ids}
    candidates = sorted(sid for sid, entry in targets.items()
                        if entry[1] == 'train' and sid not in train_ids
                        and entry[2] not in sampled_subjects
                        and (eligible is None or sid in eligible))
    if len(candidates) < int(dev_limit):
        raise ValueError(f'only {len(candidates)} patient-disjoint TRAIN rows remain for '
                         f'a {int(dev_limit)}-row dev split; lower --dev-limit or '
                         '--train-limit')
    rng = random.Random(f'dev-{sample_seed}')
    return frozenset(rng.sample(candidates, int(dev_limit)))
