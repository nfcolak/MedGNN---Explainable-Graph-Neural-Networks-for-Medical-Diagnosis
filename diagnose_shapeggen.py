import sys
sys.path.insert(0, "src")
sys.path.insert(0, "prot_gnn_core")
sys.path.insert(0, "external/GraphXAI-main")

print("=" * 60)
print("Step 1 — import ShapeGGen")
print("=" * 60)

try:
    from graphxai.datasets import ShapeGGen
    print("  OK: imported from graphxai.datasets")
except ImportError as e:
    print(f"  FAIL: {e}")
    print("\n  Trying alternate import path...")
    try:
        from graphxai.datasets.shape_graph import ShapeGGen
        print("  OK: imported from graphxai.datasets.shape_graph")
    except ImportError as e2:
        print(f"  FAIL: {e2}")
        sys.exit(1)

print()
print("=" * 60)
print("Step 2 — construct a tiny ShapeGGen dataset")
print("=" * 60)

dataset = ShapeGGen(
    num_graphs=10,
    avg_num_nodes=15,
    avg_degree=4,
    num_subgraphs=1,
    prob_connection=0.03,
    add_sensitive_feature=False,
    base_graph='ba',
    shape='house',
    seed=42,
)

print(f"  type(dataset)         : {type(dataset)}")
print(f"  type(dataset).__mro__ : {[c.__name__ for c in type(dataset).__mro__]}")

try:
    print(f"  len(dataset)          : {len(dataset)}")
except Exception as e:
    print(f"  len(dataset) FAILED   : {e}")

try:
    item = dataset[0]
    print(f"  dataset[0] type       : {type(item)}")
except Exception as e:
    print(f"  dataset[0] FAILED     : {e}")

try:
    exp = dataset.get_explanation(0)
    print(f"  get_explanation(0)    : {type(exp)}")
    print(f"  exp attrs             : {[a for a in dir(exp) if not a.startswith('_')]}")
except Exception as e:
    print(f"  get_explanation FAIL  : {e}")

print()
print("=" * 60)
print("Step 3 — inspect dataset[0] in detail")
print("=" * 60)
try:
    item = dataset[0]
    for attr in ['x', 'edge_index', 'y', 'node_mask', 'edge_mask',
                 'node_imp', 'edge_imp', 'graph_mask', 'ground_truth']:
        val = getattr(item, attr, "NOT FOUND")
        if val == "NOT FOUND":
            print(f"  {attr:<18}: NOT FOUND")
        else:
            import torch
            if torch.is_tensor(val):
                print(f"  {attr:<18}: tensor shape={tuple(val.shape)} "
                      f"dtype={val.dtype} unique={val.unique().tolist()[:6]}")
            else:
                print(f"  {attr:<18}: {val}")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=" * 60)
print("Step 4 — check if dataset is iterable / list-like")
print("=" * 60)
try:
    import numpy as np
    items = list(dataset)
    print(f"  list(dataset) length  : {len(items)}")
    print(f"  items[0] type         : {type(items[0])}")
    labels = [int(it.y.item()) for it in items]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  label distribution    : {dict(zip(unique.tolist(), counts.tolist()))}")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=" * 60)
print("Step 5 — check dataset-level attributes")
print("=" * 60)
for attr in ['num_classes', 'num_node_features', 'num_features', 'num_graphs']:
    val = getattr(dataset, attr, "NOT FOUND")
    print(f"  {attr:<22}: {val}")

print()
print("=" * 60)
print("Step 6 — inspect ShapeGGen source file")
print("=" * 60)
import inspect
print(inspect.getfile(ShapeGGen))
print()
# Print all public methods and properties
members = [(name, type(val).__name__) for name, val in inspect.getmembers(dataset)
           if not name.startswith('_')]
for name, kind in members:
    print(f"  {name:<30}: {kind}")

print()
print("=" * 60)
print("Step 7 — inspect the internal graph object")
print("=" * 60)
# NodeDataset typically stores graph as dataset.G or dataset.graph
for attr in ['G', 'graph', 'data', 'graphs', 'dataset', 'explanations',
             'node_mask', 'edge_mask', 'y', 'x', 'edge_index', 'labels']:
    val = getattr(dataset, attr, "NOT FOUND")
    if val == "NOT FOUND":
        print(f"  {attr:<20}: NOT FOUND")
    else:
        import torch
        if torch.is_tensor(val):
            print(f"  {attr:<20}: tensor shape={tuple(val.shape)} dtype={val.dtype}")
        else:
            print(f"  {attr:<20}: {type(val).__name__}  {str(val)[:80]}")

print()
print("=" * 60)
print("Step 8 — what graph classification datasets exist in GraphXAI")
print("=" * 60)
import graphxai.datasets as gxds
import os

print("All names in graphxai.datasets:")
names = [x for x in dir(gxds) if not x.startswith('_')]
print(" ", names)

print()
print("Dataset files in graphxai/datasets/:")
base = "external/GraphXAI-main/graphxai/datasets"
for f in sorted(os.listdir(base)):
    print(f"  {f}")

print()
print("Step 9 — inspect ShapeGGen.explanations[0]")
exp_list = dataset.explanations
print(f"  len(explanations)     : {len(exp_list)}")
print(f"  explanations[0] type  : {type(exp_list[0])}")
if exp_list:
    e = exp_list[0]
    if isinstance(e, list):
        print(f"  explanations[0] is a list of len {len(e)}")
        e = e[0]
    attrs = [a for a in dir(e) if not a.startswith('_')]
    print(f"  Explanation attrs     : {attrs}")
    import torch
    for attr in attrs:
        try:
            val = getattr(e, attr)
            if torch.is_tensor(val):
                print(f"    {attr:<20}: tensor shape={tuple(val.shape)} "
                      f"unique={val.unique().tolist()[:5]}")
        except Exception:
            pass

print()
print("=" * 60)
print("Step 11 — inspect ShapeGGen.explanations structure in detail")
print("=" * 60)
import torch

# We know explanations is a list of 11 items, each a list of 1 Explanation
# Check what node_idx is — this tells us which node the explanation belongs to
for i in range(min(4, len(dataset.explanations))):
    exp = dataset.explanations[i][0]
    print(f"  Node {i}: node_idx={exp.node_idx}  "
          f"node_imp unique={exp.node_imp.unique().tolist()}  "
          f"node_imp sum={exp.node_imp.sum().item():.0f}")

print()
# Check in_shape — which nodes are in the motif
print(f"  dataset.in_shape      : {dataset.in_shape}")
print(f"  dataset.graph.shape   : {dataset.graph.shape.tolist()}")
print(f"  dataset.y             : {dataset.y.tolist()}")

print()
print("Step 12 — check num_hops and model_layers on ShapeGGen")
print(f"  dataset.num_hops      : {dataset.num_hops}")
print(f"  dataset.model_layers  : {dataset.model_layers}")
print(f"  dataset.num_nodes     : {dataset.num_nodes}")
print(f"  dataset.subgraph_size : {dataset.subgraph_size}")

print()
print("Step 13 — test k_hop_subgraph extraction on node 0")
from torch_geometric.utils import k_hop_subgraph
subset, sub_edge_index, mapping, edge_mask_bool = k_hop_subgraph(
    node_idx=0,
    num_hops=dataset.num_hops,
    edge_index=dataset.edge_index,
    relabel_nodes=True,
    num_nodes=dataset.num_nodes,
)
print(f"  subset (global node ids) : {subset.tolist()}")
print(f"  subset size              : {len(subset)}")
print(f"  sub_edge_index shape     : {tuple(sub_edge_index.shape)}")
print(f"  mapping (centre local)   : {mapping}")
shape_sub = dataset.graph.shape[subset]
print(f"  shape_sub (motif mask)   : {shape_sub.tolist()}")
print(f"  motif nodes in subgraph  : {shape_sub.sum().item():.0f}")