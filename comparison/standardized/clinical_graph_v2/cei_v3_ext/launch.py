"""Training entry boundary that never deserializes held-out test graphs.

The bounded smoke uses train.py's explicit option. Full uses its unchanged
training semantics with a scoped graph reader dependency excluding test IDs;
no global monkeypatch and no temporary/rebuilt graph artifact.
"""
from __future__ import annotations

from types import FunctionType

from .. import train
from .io import selected_graphs, selected_preprocessing


def main():
    args = train.normalize_method_args(train.parser().parse_args())
    if not args.execute:
        raise ValueError('launch is an execute-only internal training boundary')
    if args.train_dev_only:
        return train.run(args)
    if (args.method != 'cei_gnn_v3' or args.selection_fold != 'dev'
            or args.final_eval != 'none' or args.min_prior_visits != 0):
        raise ValueError('extension training launch requires locked dev-selected CEI-v3')
    targets, _kept, _dropped = train.select_top_labels(train.load_targets(args.targets), 10)
    allowed = {sid for sid, entry in targets.items() if entry[1] != 'test'}
    dataset = train.build_dataset
    scope = dict(dataset.__globals__)
    scope['iter_graphs_with_membership'] = lambda graphs, members: selected_graphs(
        graphs, members, allowed)
    scope['fit_preprocessing'] = selected_preprocessing
    selected_dataset = FunctionType(dataset.__code__, scope, dataset.__name__,
                                    dataset.__defaults__, dataset.__closure__)
    selected_dataset.__kwdefaults__ = dataset.__kwdefaults__
    execute = train.run
    run_scope = dict(execute.__globals__, build_dataset=selected_dataset)
    guarded = FunctionType(execute.__code__, run_scope, execute.__name__,
                           execute.__defaults__, execute.__closure__)
    return guarded(args)


if __name__ == '__main__':
    main()
