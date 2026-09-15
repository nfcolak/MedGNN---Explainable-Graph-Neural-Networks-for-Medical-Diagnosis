"""Pure policy helpers for isolated, validation-only performance experiments."""
import numpy as np


def class_weights(labels, num_classes, policy):
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError('All canonical classes must occur in training')
    weights = len(labels) / (num_classes * counts)
    if policy == 'inverse':
        return weights
    if policy == 'sqrt_inverse':
        return np.sqrt(weights)
    if policy == 'none':
        return np.ones(num_classes)
    raise ValueError('Unknown weighting policy')


def graph_vector(x, concept_start, concept_end, hub_start, view):
    concepts = np.clip(x[:, concept_start:concept_end].sum(axis=0), 0, 1)
    if view == 'concepts':
        return concepts
    if view == 'native':
        return np.concatenate([concepts, x[0, hub_start:]])
    raise ValueError('Unknown input view')


def best_validation(rows):
    return max(rows, key=lambda row: row['macro_f1'])


def validate_budget(method, variant, epochs, seed):
    if (method not in ('protgnn', 'gsat', 'graphcare')
            or variant not in ('current', 'sqrt_inverse')
            or type(epochs) is not int or not 1 <= epochs <= 3
            or seed != 1234):
        raise ValueError('Budget: three methods, two losses, at most three epochs, seed 1234')
