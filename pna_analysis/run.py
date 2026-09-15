"""Opt-in bounded PNA source-snapshot smoke runner; never the legacy benchmark matrix."""
import argparse
import json
from pathlib import Path

from pna_analysis.data import load_common_input
from pna_analysis.model import capacity_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / 'comparison/standardized/pna_experiments'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, default=PROJECT_ROOT / 'comparison/standardized/common_input_20260913')
    p.add_argument('--source', type=Path, default=PROJECT_ROOT / 'data/merged_ed.csv')
    p.add_argument('--split', type=Path, default=PROJECT_ROOT / 'comparison/canonical_split.json')
    p.add_argument('--output-dir', type=Path, help='NEW directory under comparison/standardized/pna_experiments')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true', help='Validate existing input and print config only (default)')
    action.add_argument('--execute', action='store_true', help='Explicitly enable bounded training')
    action.add_argument('--replay', type=Path, help='Read-only replay of a completed PNA run directory')
    p.add_argument('--architecture', choices=['interaction', 'plain', 'plain_wide'], default='interaction')
    p.add_argument('--width', type=int, choices=range(8, 129), default=64, metavar='8..128')
    p.add_argument('--layers', type=int, choices=range(1, 4), default=2)
    p.add_argument('--rank', type=int, choices=range(1, 33), default=16, metavar='1..32')
    p.add_argument('--disable-cross-pairs', action='store_true', help='Same-parameter interaction ablation')
    p.add_argument('--no-messages', action='store_true', help='Zero every real edge; diagnostic negative control')
    p.add_argument('--loss', choices=['ce', 'sqrt_inverse'], default='ce')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--steps', type=int, choices=range(1, 101), default=8, metavar='1..100')
    p.add_argument('--train-limit', type=int, choices=range(1, 4097), default=256, metavar='1..4096')
    p.add_argument('--val-limit', type=int, choices=range(0, 1025), default=64, metavar='0..1024',
                   help='0 evaluates the train smoke subset; never builds a test loader')
    p.add_argument('--batch-size', type=int, choices=range(1, 129), default=32, metavar='1..128')
    return p


def make_plan(args, bundle):
    contract = bundle['contract']
    if args.train_limit > contract['counts'][0] or args.val_limit > contract['counts'][1]:
        raise ValueError('Smoke subset exceeds corresponding fold')
    capacities = capacity_report(len(contract['names']) + 1, len(contract['classes']), args.width, args.layers, args.rank)
    config = {'architecture': args.architecture, 'width': capacities[args.architecture]['width'],
              'reference_width': args.width, 'layers': args.layers, 'rank': args.rank,
              'interactions': args.architecture == 'interaction', 'cross_pairs': not args.disable_cross_pairs,
              'no_messages': args.no_messages, 'seed': args.seed, 'steps': args.steps,
              'train_limit': args.train_limit, 'val_limit': args.val_limit, 'batch_size': args.batch_size,
              'loss': args.loss, 'optimizer': 'Adam', 'lr': .001, 'weight_decay': 1e-5,
              'device': 'cpu', 'threads': 4, 'dropout': 0., 'normalization': 'none',
              'aggregators': ['mean', 'min', 'max', 'std'],
              'scalers': ['identity', 'amplification', 'attenuation'],
              'towers': 1, 'pre_layers': 1, 'post_layers': 1, 'readout': 'hub_only',
              'residual': True, 'degree_fit': 'all canonical train graphs only; detached',
              'class_weight_fit': 'all canonical train labels only',
              'selection': 'fixed-step last checkpoint; no validation selection',
              'primary_metric': 'macro_f1', 'metric_implementation': 'shared.lib.metrics.multiclass_metrics',
              'subset_policy': 'seeded train permutation; first validation rows in unchanged NPZ order'}
    return {'action': 'execute' if args.execute else 'dry_run', 'scope': 'bounded wiring smoke, not performance evidence',
            'config': config, 'capacities': capacities, 'input_counts': contract['counts'],
            'run_bindings_verified': bundle['run_bindings_verified'],
            'test_evaluated': False, 'temporal_clean': False, 'raw_to_model_train_only': False,
            'input_paths': {'root': str(args.input_root.resolve()), 'source': str(args.source.resolve()), 'split': str(args.split.resolve())},
            'output_dir': str(args.output_dir.resolve()) if args.output_dir else None}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.replay is not None:
        from pna_analysis.train import verify_checkpoint
        manifest = json.loads((args.replay / 'run_manifest.json').read_text())
        if manifest['status'] != 'completed':
            raise ValueError('Replay requires a completed run')
        paths = manifest['input_paths']
        bundle = load_common_input(paths['root'], paths['source'], paths['split'])
        print(json.dumps(verify_checkpoint(args.replay, bundle), sort_keys=True))
        return 0
    bundle = load_common_input(args.input_root, args.source, args.split)
    plan = make_plan(args, bundle)
    if not args.execute:
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise ValueError('--execute requires explicit --output-dir')
    target = args.output_dir.resolve()
    if EXPERIMENT_ROOT.resolve() not in target.parents:
        raise ValueError('Output must be a NEW child of comparison/standardized/pna_experiments')
    from pna_analysis.train import train_smoke
    manifest = train_smoke(plan, bundle, target)
    print(json.dumps({'status': manifest['status'], 'optimizer_steps': manifest['optimizer_steps'],
                      'output_dir': str(target), 'replay': manifest['replay']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
