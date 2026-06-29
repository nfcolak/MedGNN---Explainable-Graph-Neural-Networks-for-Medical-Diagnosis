import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _path in (_PROJECT_ROOT, _SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from protgnn_analysis.config import data_args
from protgnn_analysis.load_dataset import get_dataset

dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
data = dataset[0]
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features per node: {data.num_node_features}")
print(f"Label: {data.y}")
