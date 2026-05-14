from load_dataset import get_dataset
dataset = get_dataset('./datasets', 'mimic')
data = dataset[0]
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features per node: {data.num_node_features}")
print(f"Label: {data.y}")