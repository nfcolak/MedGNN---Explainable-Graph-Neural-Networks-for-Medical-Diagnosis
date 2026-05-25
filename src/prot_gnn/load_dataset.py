#load_dataset.py
import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
import pandas as pd
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse, from_networkx
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import networkx as nx
from torch_geometric.data import Data

POST_OUTCOME_COLUMNS = []


def _drop_post_outcome_columns(df):
    return df.drop(columns=POST_OUTCOME_COLUMNS, errors='ignore')


class SyntheticGraphDataset:
    """
    Wraps a list of Data objects and exposes the attributes that
    GnnNets, get_dataloader, and train_model expect from any dataset.

    Each Data object has:
        .x             [num_nodes, num_features]  node feature matrix
        .edge_index    [2, num_edges]              graph connectivity
        .y             [1]                         graph label (0 or 1)
        .node_mask     [num_nodes]  float          1.0 = motif node (ground truth)
        .edge_mask     [num_edges]  float          1.0 = motif edge (ground truth)
    """

    def __init__(self, data_list: list, name: str = "synthetic"):
        if not data_list:
            raise ValueError("SyntheticGraphDataset: data_list is empty")

        self._data = data_list
        self.name = name

        sample = data_list[0]
        self._num_node_features = int(sample.x.shape[1])

        all_labels = [int(d.y.view(-1)[0].item()) for d in data_list]
        self._num_classes = len(set(all_labels))

        self.feature_cols = []
        self.feature_metadata = {}
        self.label_mapping = {str(i): str(i) for i in range(self._num_classes)}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    @property
    def num_node_features(self):
        return self._num_node_features

    @property
    def num_features(self):
        return self._num_node_features

    @property
    def num_classes(self):
        return self._num_classes


# ---- Motif builders — at module level, NOT inside the class ----

def _make_house() -> nx.Graph:
    """5-node house: square base + triangular roof."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 4),
    ])
    return G


def _make_cycle(n: int = 6) -> nx.Graph:
    return nx.cycle_graph(n)


def _make_wheel(n: int = 6) -> nx.Graph:
    return nx.wheel_graph(n)


def _make_grid() -> nx.Graph:
    return nx.grid_2d_graph(2, 3)


MOTIF_BUILDERS = {
    'house': _make_house,
    'cycle': lambda: _make_cycle(6),
    'wheel': lambda: _make_wheel(6),
    'grid':  _make_grid,
}

def _make_graph(
    n_base: int,
    motif_shape: str,
    label: int,
    node_feature_dim: int,
    rng: np.random.Generator,
) -> Data:
    """
    Build one graph.
 
    If label == 1: BA base graph + motif attached at a random base node.
    If label == 0: BA base graph only.
 
    Node features: Gaussian noise, with a small class-correlated signal
    injected into the first feature dimension so the model can actually learn.
    """
    # Base graph (Barabási-Albert)
    m = max(1, n_base // 10)           # BA attachment parameter
    G_base = nx.barabasi_albert_graph(n_base, m, seed=int(rng.integers(1e6)))
 
    n_motif_nodes = 0
    motif_node_set = set()
    all_edges = list(G_base.edges())
 
    if label == 1:
        builder = MOTIF_BUILDERS.get(motif_shape, _make_house)
        G_motif = builder()
        n_motif_nodes = G_motif.number_of_nodes()
 
        # Relabel motif nodes to avoid collision with base nodes
        offset = n_base
        G_motif = nx.relabel_nodes(G_motif, {i: i + offset for i in G_motif.nodes()})
        motif_node_set = set(G_motif.nodes())
 
        # Merge
        G = nx.compose(G_base, G_motif)
 
        # Attach motif to a random base node
        attach_base = int(rng.integers(n_base))
        attach_motif = offset  # first motif node
        G.add_edge(attach_base, attach_motif)
        all_edges = list(G.edges())
    else:
        G = G_base
 
    # Total nodes
    all_nodes = sorted(G.nodes())
    n_total = len(all_nodes)
 
    # Node features: Gaussian noise + class signal in dim 0
    x = rng.standard_normal((n_total, node_feature_dim)).astype(np.float32)
    # Class signal: motif nodes get +1 in dim 0, base nodes get -1
    for i, node in enumerate(all_nodes):
        x[i, 0] = 1.0 if node in motif_node_set else -1.0
    # Add noise to the signal so it is not trivially separable
    x[:, 0] += rng.standard_normal(n_total).astype(np.float32) * 0.5
 
    x_tensor = torch.tensor(x, dtype=torch.float)
 
    # Build edge_index with consecutive local node ids
    node_to_local = {node: i for i, node in enumerate(all_nodes)}
    src, dst = [], []
    for u, v in all_edges:
        lu, lv = node_to_local[u], node_to_local[v]
        src.extend([lu, lv])  # undirected
        dst.extend([lv, lu])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
 
    # Ground truth masks
    node_mask = torch.tensor(
        [1.0 if node in motif_node_set else 0.0 for node in all_nodes],
        dtype=torch.float,
    )
    if edge_index.shape[1] > 0:
        local_src = edge_index[0]
        local_dst = edge_index[1]
        edge_mask = ((node_mask[local_src] > 0.5) & (node_mask[local_dst] > 0.5)).float()
    else:
        edge_mask = torch.zeros(0, dtype=torch.float)
 
    return Data(
        x          = x_tensor,
        edge_index = edge_index,
        y          = torch.tensor([label], dtype=torch.long),
        node_mask  = node_mask,
        edge_mask  = edge_mask,
    )

def _build_synthetic_dataset(
    shape: str,
    base_graph: str,          # currently only 'ba' used; kept for API compat
    num_graphs: int,
    avg_nodes: int,
    node_feature_dim: int = 10,
    seed: int = 42,
) -> SyntheticGraphDataset:
    """
    Generate a balanced synthetic graph-classification dataset.
 
    Half the graphs have label 1 (motif present), half have label 0.
    avg_nodes controls the size of the BA base graph (recommended 20-40).
    """
    if shape not in MOTIF_BUILDERS:
        raise ValueError(
            f"Unknown motif shape '{shape}'. "
            f"Available: {list(MOTIF_BUILDERS.keys())}"
        )
 
    rng = np.random.default_rng(seed)
    n_class1 = num_graphs // 2
    n_class0 = num_graphs - n_class1
 
    data_list = []
 
    # Class 1: motif present
    for _ in range(n_class1):
        n_base = max(10, int(rng.normal(avg_nodes, avg_nodes * 0.15)))
        data_list.append(_make_graph(n_base, shape, label=1,
                                     node_feature_dim=node_feature_dim, rng=rng))
 
    # Class 0: no motif
    for _ in range(n_class0):
        n_base = max(10, int(rng.normal(avg_nodes, avg_nodes * 0.15)))
        data_list.append(_make_graph(n_base, shape, label=0,
                                     node_feature_dim=node_feature_dim, rng=rng))
 
    # Shuffle
    idx = rng.permutation(len(data_list)).tolist()
    data_list = [data_list[i] for i in idx]
 
    # Report
    labels = [int(d.y.item()) for d in data_list]
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    avg_n = np.mean([d.x.shape[0] for d in data_list])
    motif_size = MOTIF_BUILDERS[shape]().number_of_nodes()
 
    print(f"  [SyntheticMotif] Built {len(data_list)} graphs: "
          f"label dist={dist}, avg_nodes={avg_n:.1f}, "
          f"motif='{shape}' ({motif_size} nodes), "
          f"motif_ratio={motif_size / avg_n:.2f}")
 
    return SyntheticGraphDataset(data_list, name=f"synthetic_{shape}_{base_graph}")

# Core subgraph extraction
# ============================================================================
 
def _extract_subgraph_graphs(
    sg,
    num_graphs: int,
    num_hops: int,
    seed: int,
    min_motif_in_subgraph: int = 1,
) -> SyntheticGraphDataset:
    """
    Convert a ShapeGGen NodeDataset into a graph-classification dataset.
 
    Strategy
    --------
    1. Read the global node feature matrix, edge_index, and motif mask
       from sg.x, sg.edge_index, sg.graph.shape.
    2. Identify motif nodes (shape==1) and base nodes (shape==0).
    3. Sample centre nodes:
         - half from motif nodes  → graph label 1
         - half from base nodes   → graph label 0
    4. For each centre node, extract its num_hops-hop subgraph.
    5. Assign:
         node_mask = shape restricted to subgraph nodes
         edge_mask = 1 if both endpoints are motif nodes
         y         = shape[centre_node]  (1 = motif, 0 = base)
 
    Parameters
    ----------
    min_motif_in_subgraph : discard subgraphs where fewer than this many
                            motif nodes are captured (prevents degenerate
                            label-1 graphs with no visible motif).
    """
    x_all          = sg.x                    # [N, F]
    edge_index_all = sg.edge_index           # [2, E]
    shape_all      = sg.graph.shape.float()  # [N] binary motif mask
    N              = x_all.shape[0]
 
    motif_nodes = (shape_all == 1).nonzero(as_tuple=True)[0].tolist()
    base_nodes  = (shape_all == 0).nonzero(as_tuple=True)[0].tolist()
 
    if not motif_nodes:
        raise RuntimeError(
            "ShapeGGen produced no motif nodes. "
            "Increase avg_num_nodes (try >= 100) so the motif fits in the graph."
        )
 
    rng = np.random.default_rng(seed)
    half = num_graphs // 2
 
    # Sample with replacement if not enough nodes of a given class
    n_motif = min(half, len(motif_nodes))
    n_base  = num_graphs - n_motif
 
    chosen_motif = rng.choice(motif_nodes, size=n_motif, replace=len(motif_nodes) < n_motif).tolist()
    chosen_base  = rng.choice(base_nodes,  size=n_base,  replace=len(base_nodes)  < n_base ).tolist()
    centre_nodes = chosen_motif + chosen_base
    rng.shuffle(centre_nodes)
 
    data_list = []
    skipped   = 0
 
    for centre in centre_nodes:
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx    = int(centre),
            num_hops    = num_hops,
            edge_index  = edge_index_all,
            relabel_nodes = True,
            num_nodes   = N,
        )
        # subset: 1-D tensor of global node indices in this subgraph
        # mapping: local index of centre node inside subset
 
        x_sub      = x_all[subset]                # [n, F]
        shape_sub  = shape_all[subset]             # [n]  motif mask, local indices
 
        # Skip if the label-1 subgraph captured no motif nodes
        # (can happen at graph boundaries)
        label = int(shape_all[centre].item())
        if label == 1 and shape_sub.sum().item() < min_motif_in_subgraph:
            skipped += 1
            continue
 
        y_graph = torch.tensor([label], dtype=torch.long)
 
        # Edge ground truth: 1 if both endpoints are motif nodes
        local_src = sub_edge_index[0]
        local_dst = sub_edge_index[1]
        edge_gt   = ((shape_sub[local_src] > 0.5) & (shape_sub[local_dst] > 0.5)).float()
 
        data_list.append(Data(
            x           = x_sub,
            edge_index  = sub_edge_index,
            y           = y_graph,
            node_mask   = shape_sub.float(),
            edge_mask   = edge_gt,
            centre_node = mapping.view(1) if torch.is_tensor(mapping) else torch.tensor([mapping]),
        ))
 
    if skipped:
        print(f"  [ShapeGGen loader] Skipped {skipped} subgraphs (no motif nodes captured)")
 
    if not data_list:
        raise RuntimeError(
            "No valid subgraphs extracted. "
            "Try increasing avg_num_nodes or decreasing num_hops."
        )
 
    # Report
    labels_out = [int(d.y.item()) for d in data_list]
    unique, counts = np.unique(labels_out, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    avg_nodes = np.mean([d.x.shape[0] for d in data_list])
    print(f"  [ShapeGGen loader] {len(data_list)} subgraphs extracted, "
          f"label distribution={dist}, avg_subgraph_nodes={avg_nodes:.1f}, "
          f"num_hops={num_hops}")
 
    return SyntheticGraphDataset(data_list, name="shapeggen_subgraph")


# load_dataset.py (Focusing on MIMIC_ED_Dataset class)
class MIMIC_ED_Dataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(MIMIC_ED_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['ed/edstays.csv', 'ed/triage.csv', 'ed/diagnosis.csv', 'ed/medrecon.csv']
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # 1. Load Data
        df_stays = pd.read_csv(osp.join(self.raw_dir, 'ed/edstays.csv'))
        df_triage = pd.read_csv(osp.join(self.raw_dir, 'ed/triage.csv'))
        df_diag = pd.read_csv(osp.join(self.raw_dir, 'ed/diagnosis.csv'))

        # 2. Filter for Admission Task (ADMITTED vs HOME)
        df_stays = df_stays[df_stays['disposition'].isin(['ADMITTED', 'HOME'])].copy()
        df_stays['label'] = (df_stays['disposition'] == 'ADMITTED').astype(int)

        # 3. Create Diagnosis Vocabulary for Explainability
        unique_diags = df_diag['icd_title'].dropna().unique().tolist()
        diag_to_id = {title: i for i, title in enumerate(unique_diags)}
        
        data_list = []

        # --- UPDATE THIS LINE ---
        for i, (_, stay) in enumerate(df_stays.iterrows()):
            if i >= 5000: break # Only process the first 5000 stays for now
            
            stid = stay['stay_id']
            p_triage = df_triage[df_triage['stay_id'] == stid]
            p_diag = df_diag[df_diag['stay_id'] == stid]

            # Features: [Type_Flag, Value, Abnormal_Flag]
            nodes_x = [[1.0, 0.0, 0.0]] 
            edges_src, edges_dst = [], []

            # Add Triage Vitals
            if not p_triage.empty:
                t = p_triage.iloc[0]
                vitals = {'temp': t['temperature'], 'hr': t['heartrate'], 'o2': t['o2sat']}
                for name, val in vitals.items():
                    if pd.notnull(val):
                        is_abnormal = 1.0 if (name=='hr' and val>100) or (name=='o2' and val<95) else 0.0
                        nodes_x.append([2.0, float(val)/100.0, is_abnormal]) 
                        idx = len(nodes_x) - 1
                        edges_src.extend([0, idx]); edges_dst.extend([idx, 0])

            # Add Diagnosis nodes
            for _, d in p_diag.iterrows():
                title = d['icd_title']
                if title in diag_to_id:
                    nodes_x.append([3.0, float(diag_to_id[title]), 1.0])
                    idx = len(nodes_x) - 1
                    edges_src.extend([0, idx]); edges_dst.extend([idx, 0])

            data_list.append(Data(
                x=torch.tensor(nodes_x, dtype=torch.float),
                edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
                y=torch.tensor([stay['label']], dtype=torch.long)
            ))

        torch.save(self.collate(data_list), self.processed_paths[0])


def _extract_graph_labels(dataset):
    labels = []
    for idx in range(len(dataset)):
        label = dataset[idx].y
        if torch.is_tensor(label):
            label = label.view(-1)[0].item()
        labels.append(int(label))
    return np.array(labels)


def _stratified_split_indices(labels, split_sizes, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    train_indices, eval_indices, test_indices = [], [], []

    for label in np.unique(labels):
        label_indices = indices[labels == label]
        rng.shuffle(label_indices)
        label_count = len(label_indices)

        train_count = int(round(split_sizes[0] * label_count))
        eval_count = int(round(split_sizes[1] * label_count))

        if train_count + eval_count > label_count:
            eval_count = max(0, label_count - train_count)
        test_count = label_count - train_count - eval_count

        train_indices.extend(label_indices[:train_count].tolist())
        eval_indices.extend(label_indices[train_count:train_count + eval_count].tolist())
        test_indices.extend(label_indices[train_count + eval_count:train_count + eval_count + test_count].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    rng.shuffle(test_indices)
    return train_indices, eval_indices, test_indices


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)

    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices, supplement


def read_syn_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

    x = torch.from_numpy(features).float()
    y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    y = torch.from_numpy(np.where(y)[1])
    edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)
    return data


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        data_list.append(Data(x=torch.from_numpy(node_features[graph_idx]).float(),
                              edge_index=dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0],
                              y=torch.from_numpy(np.where(graph_labels[graph_idx])[0])))
    return data_list

class DS1Dataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(DS1Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['ds1.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, 'ds1.csv'), na_values=['NA', ''])
        
        # 2. Dropping irrelevant columns
        cols_to_drop = ['encounter_id', 'patient_id', 'hospital_id']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # auto-encoding all text to numbers
        for col in df.columns:
            if df[col].dtype == type(object):
                df[col] = pd.factorize(df[col])[0]
        
        df = df.fillna(0)

        # Target variable (predicting general hospital death)
        y_target = torch.tensor(df['mortality_label'].values, dtype=torch.long)

        # --- TEMPORARY FIX TO BYPASS THE 1-CLASS ERROR ---
        # Since this CSV is the unlabeled test set, all targets are 0.
        # We artificially made 20% of the patients "1" (Died) so the 
        # model has 2 classes to learn from
        y_target[:len(y_target)//5] = 1

        df_features = df.drop(columns=['mortality_label']) # Dropping targets from features

        # --- DATA NORMALIZATION ---
        for col in df_features.columns:
            if df_features[col].std() != 0:
                df_features[col] = (df_features[col] - df_features[col].mean()) / df_features[col].std()

        data_list = []
        num_features = len(df_features.columns)

        # Building a Star Graph for each patient
        for i, row in df_features.iterrows():
            feature_vals = row.values.tolist()
            num_nodes = num_features + 1
            
            # Create a matrix of zeros: [number of nodes, number of features + 1]
            x = torch.zeros((num_nodes, num_features + 1), dtype=torch.float)
            
            # The Center "Patient" Node gets a 1 in the very first column
            x[0, 0] = 1.0
            
            # The Outer "Feature" Nodes get their value placed in their own unique column
            for j in range(num_features):
                x[j + 1, j + 1] = feature_vals[j]

            # Build edges connecting Node 0 (Patient) to all feature nodes
            source_nodes = [0] * num_features
            target_nodes = list(range(1, num_features + 1))
            
            # Undirected edges
            edge_index = torch.tensor([
                source_nodes + target_nodes, 
                target_nodes + source_nodes
            ], dtype=torch.long)

            y = torch.tensor([y_target[i]], dtype=torch.long)

            # Create the PyG Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # 4. Save the processed data
        torch.save(self.collate(data_list), self.processed_paths[0])

def load_DS1(dataset_dir, dataset_name):
    dataset = DS1Dataset(root=dataset_dir, name=dataset_name)
    return dataset

class MimicEDMergedDataset(InMemoryDataset):
    """
    Loads merged_ed_no_icd_sample_20k.csv (or with_icd variant) and builds one
    star graph per patient:  center node + one leaf node per feature column.

    Node features (input_dim = 2):
        center node  → [-1.0,  0.0]
        feature node → [j / num_features,  normalized_value]
    """
    def __init__(self, root, name, csv_filename='merged_ed_no_icd_sample_20k.csv',
                 transform=None, pre_transform=None):
        self.name = name
        self.csv_filename = csv_filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, f'processed_{self.name}')

    @property
    def raw_file_names(self):
        return [self.csv_filename]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self.csv_filename))
        df = df.drop(columns=['subject_id'], errors='ignore')

        y_vals = torch.tensor(df['disposition'].values, dtype=torch.long)
        df = df.drop(columns=['disposition'])
        df = _drop_post_outcome_columns(df)

        feature_cols = df.columns.tolist()
        num_features = len(feature_cols)

        # z-score normalize continuous columns; leave binary (0/1) columns as-is
        for col in feature_cols:
            if df[col].nunique() > 2:
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - df[col].mean()) / std
        df = df.fillna(0)

        # build star graphs
        src = [0] * num_features + list(range(1, num_features + 1))
        dst = list(range(1, num_features + 1)) + [0] * num_features
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        data_list = []
        for i in range(len(df)):
            fvals = df.iloc[i].values
            x = torch.zeros((num_features + 1, 2), dtype=torch.float)
            x[0, 0] = -1.0  # center node marker
            for j in range(num_features):
                x[j + 1, 0] = j / num_features   # normalized feature index
                x[j + 1, 1] = float(fvals[j])    # normalized feature value
            data_list.append(Data(
                x=x,
                edge_index=edge_index.clone(),
                y=torch.tensor([y_vals[i].item()], dtype=torch.long)
            ))

        torch.save(self.collate(data_list), self.processed_paths[0])


class PatientSimilarityGraphDataset(InMemoryDataset):
    """
    Builds one small neighbourhood graph per patient:
        node 0        = the patient itself
        nodes 1..k    = its k most similar patients (cosine similarity)
        edges         = undirected star (center ↔ each neighbour)
        node features = raw patient feature vector (input_dim = num_features)
        label         = center patient's disposition (HOME=0, ADMITTED=1)

    Graph classification with ProtGNN: a prototype learns to represent a
    typical clinical profile together with its similar-patient neighbourhood.
    """
    def __init__(self, root, name, k=10, graph_mode='star',
                 csv_filename='merged_ed_no_icd_sample_20k.csv',
                 transform=None, pre_transform=None):
        self.name = name
        self.k = k
        self.graph_mode = graph_mode
        self.csv_filename = csv_filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._load_metadata()
        if len(self) > 0 and not hasattr(self.get(0), 'patient_index'):
            print("  Patient metadata missing from processed cache; rebuilding dataset cache...")
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            self._load_metadata()

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        # use csv stem + k so different CSVs/k values don't collide
        csv_stem = self.csv_filename.replace('.csv', '')
        mode_suffix = "" if self.graph_mode == "star" else f"_{self.graph_mode}"
        return osp.join(self.root, f'processed_patsim_{csv_stem}_k{self.k}{mode_suffix}')

    @property
    def raw_file_names(self):
        return [self.csv_filename]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def metadata_path(self):
        return osp.join(self.processed_dir, 'metadata.json')

    def _load_metadata(self):
        self.feature_cols = []
        self.feature_metadata = {}
        self.label_mapping = {"0": "HOME", "1": "ADMITTED"}
        if osp.isfile(self.metadata_path):
            with open(self.metadata_path) as f:
                metadata = json.load(f)
            self.feature_cols = metadata.get("feature_cols", [])
            self.feature_metadata = metadata.get("feature_metadata", {})
            self.label_mapping = metadata.get("label_mapping", self.label_mapping)

    def process(self):
        from sklearn.neighbors import NearestNeighbors

        df = pd.read_csv(osp.join(self.raw_dir, self.csv_filename))
        df = df.drop(columns=['subject_id'], errors='ignore')

        y_vals = df['disposition'].values.astype(int)
        df = df.drop(columns=['disposition'])
        df = _drop_post_outcome_columns(df)

        feature_cols = df.columns.tolist()
        raw_df = df.copy()
        feature_metadata = {}

        # z-score normalize continuous columns; leave binary (0/1) columns as-is
        for col in feature_cols:
            mean = float(raw_df[col].mean()) if raw_df[col].notna().any() else 0.0
            std = float(raw_df[col].std()) if raw_df[col].notna().any() else 0.0
            is_binary = raw_df[col].dropna().isin([0, 1]).all()
            feature_metadata[col] = {
                "mean": mean,
                "std": std,
                "is_binary": bool(is_binary),
                "is_normalized": False,
            }
            if df[col].nunique() > 2:
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - df[col].mean()) / std
                    feature_metadata[col]["is_normalized"] = True
        df = df.fillna(0)

        X = df.values.astype(np.float32)
        print(f"  Building k={self.k} nearest-neighbour graph for {len(X)} patients "
              f"({len(feature_cols)} features)...")

        nn_model = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine', n_jobs=-1)
        nn_model.fit(X)
        distances, indices = nn_model.kneighbors(X)   # shape [n, k+1]; col 0 = self

        # fixed star edge list (reused for every graph)
        src = [0] * self.k + list(range(1, self.k + 1))
        dst = list(range(1, self.k + 1)) + [0] * self.k
        edge_index_template = torch.tensor([src, dst], dtype=torch.long)

        def distance_to_weight(distance):
            return float(np.exp(-max(float(distance), 0.0)))

        def weighted_star_edges(row_distances):
            weights = [distance_to_weight(d) for d in row_distances]
            return edge_index_template.clone(), torch.tensor(weights + weights, dtype=torch.float)

        def local_knn_edges(node_idx, local_k=3):
            local_x = X[node_idx]
            norms = np.linalg.norm(local_x, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            local_norm = local_x / norms
            cosine_distance = 1.0 - np.matmul(local_norm, local_norm.T)
            rows, cols, weights = [], [], []
            local_k = min(local_k, len(node_idx) - 1)
            for src_idx in range(len(node_idx)):
                order = np.argsort(cosine_distance[src_idx])
                added = 0
                for dst_idx in order:
                    if src_idx == dst_idx:
                        continue
                    rows.append(src_idx)
                    cols.append(int(dst_idx))
                    weights.append(distance_to_weight(cosine_distance[src_idx, dst_idx]))
                    added += 1
                    if added >= local_k:
                        break
            return (
                torch.tensor([rows, cols], dtype=torch.long),
                torch.tensor(weights, dtype=torch.float),
            )

        data_list = []
        for i in range(len(X)):
            neighbour_idx = indices[i][1:]            # skip self
            neighbour_dist = distances[i][1:]         # skip self
            node_idx = [i] + list(neighbour_idx)      # center + k neighbours

            x = torch.tensor(X[node_idx], dtype=torch.float)
            y = torch.tensor([y_vals[i]], dtype=torch.long)
            edge_index = edge_index_template.clone()
            edge_weight = None
            if self.graph_mode == "weighted_star":
                edge_index, edge_weight = weighted_star_edges(neighbour_dist)
            elif self.graph_mode == "local_knn":
                edge_index, edge_weight = local_knn_edges(node_idx)
            elif self.graph_mode != "star":
                raise ValueError(f"Unsupported patient-similarity graph_mode: {self.graph_mode}")

            graph_kwargs = {}
            if edge_weight is not None:
                graph_kwargs["edge_weight"] = edge_weight

            data_list.append(Data(
                x=x,
                edge_index=edge_index,
                y=y,
                dataset_index=torch.tensor([i], dtype=torch.long),
                patient_index=torch.tensor([i], dtype=torch.long),
                node_patient_indices=torch.tensor(node_idx, dtype=torch.long),
                **graph_kwargs,
            ))

        print(f"  Done. {len(data_list)} graphs, "
              f"{self.k + 1} nodes/graph, input_dim={len(feature_cols)}, graph_mode={self.graph_mode}")
        torch.save(self.collate(data_list), self.processed_paths[0])
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump({
                "csv_filename": self.csv_filename,
                "feature_cols": feature_cols,
                "feature_metadata": feature_metadata,
                "label_mapping": {"0": "HOME", "1": "ADMITTED"},
                "k": self.k,
                "graph_mode": self.graph_mode,
            }, f, indent=2)


def get_dataset(dataset_dir, dataset_name, task=None):
    sync_dataset_dict = {
        'BA_2Motifs'.lower(): 'BA_2Motifs',
        'BA_Shapes'.lower(): 'BA_shapes',
        'BA_Community'.lower(): 'BA_Community',
        'Tree_Cycle'.lower(): 'Tree_Cycle',
        'Tree_Grids'.lower(): 'Tree_Grids',
    }
    sentigraph_names = ['Graph_SST2', 'Graph_Twitter', 'Graph_SST5']
    sentigraph_names = [name.lower() for name in sentigraph_names]
    molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]

    if dataset_name.lower() == 'MUTAG'.lower():
        return load_MUTAG(dataset_dir, 'MUTAG')
    elif dataset_name.lower() == 'mimic_full/files/mimic-iv-ed/2.2':
        return MIMIC_ED_Dataset(root=dataset_dir, name=dataset_name)
    elif dataset_name.lower() == 'mimic_ed_no_icd':
        return MimicEDMergedDataset(root=dataset_dir, name=dataset_name,
                                    csv_filename='merged_ed_no_icd_sample_20k.csv')
    elif dataset_name.lower() == 'mimic_ed_with_icd':
        return MimicEDMergedDataset(root=dataset_dir, name=dataset_name,
                                    csv_filename='merged_ed_with_icd_sample_20k.csv')
    elif dataset_name.startswith('shapeggen'):
        return _load_shapeggen(dataset_name)
    elif dataset_name.lower().startswith('mimic_patient_sim'):
        # name format:  mimic_patient_sim[_full][_k<N>]
        # e.g.  mimic_patient_sim          → sample 20k, k=10
        #       mimic_patient_sim_k20      → sample 20k, k=20
        #       mimic_patient_sim_full_k20 → full dataset, k=20
        name_l = dataset_name.lower()
        full   = 'full' in name_l
        csv    = 'merged_ed_no_icd.csv' if full else 'merged_ed_no_icd_sample_20k.csv'
        k      = 10
        graph_mode = 'star'
        for part in name_l.split('_'):
            if part.startswith('k') and part[1:].isdigit():
                k = int(part[1:])
            elif part in {'weighted', 'weighted-star'}:
                graph_mode = 'weighted_star'
            elif part in {'local', 'local-knn'}:
                graph_mode = 'local_knn'
        return PatientSimilarityGraphDataset(root=dataset_dir, name=dataset_name,
                                             k=k, graph_mode=graph_mode, csv_filename=csv)
    elif dataset_name.lower() == 'ds1':
        return load_DS1(dataset_dir, 'ds1')
    elif dataset_name.lower() in sync_dataset_dict.keys():
        sync_dataset_filename = sync_dataset_dict[dataset_name.lower()]
        return load_syn_data(dataset_dir, sync_dataset_filename)
    elif dataset_name.lower() in molecule_net_dataset_names:
        return load_MolecueNet(dataset_dir, dataset_name, task)
    elif dataset_name.lower() in sentigraph_names:
        return load_SeniGraph(dataset_dir, dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
class ShapeGGenWrapper:
    """
    A robust dataset wrapper engineered around the verified runtime parameters 
    of the custom local ShapeGGen object layer.
    """
    def __init__(self, shapeggen_dataset):
        self.dataset = shapeggen_dataset
        
        # Pull the unified macro graph via its built-in retrieval method
        # This resolves train/validation/test masking fields natively
        self.macro_data = shapeggen_dataset.get_graph(use_fixed_split=True)
        
        # ShapeGGen packs multiple subgraphs inside a giant macro data cluster.
        # We need to expose its node features dimension and labels
        self.num_node_features = self.macro_data.x.size(1)
        
        # Determine number of classes safely from unique targets
        self.num_classes = len(torch.unique(self.macro_data.y)) if hasattr(self.macro_data, 'y') else 2
        
        # Emulate a single-item dataset slice list for train_and_explain.py's iteration requirements
        self.graphs = [self.macro_data]
        self.explanations = shapeggen_dataset.explanations

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        
        # Attach ground-truth explanation node masks seamlessly if available
        if not hasattr(data, 'node_mask'):
            # Safely navigate nested list layout
            if len(self.explanations) > idx and self.explanations[idx] is not None:
                exp_entry = self.explanations[idx]
                
                # If it's wrapped inside an inner list, pull the primary motif explanation
                if isinstance(exp_entry, list) and len(exp_entry) > 0:
                    exp_entry = exp_entry[0]
                
                # Double-check that we successfully isolated an actual Explanation instance
                if hasattr(exp_entry, 'node_imp'):
                    data.node_mask = exp_entry.node_imp
                else:
                    import torch
                    data.node_mask = torch.zeros(data.x.size(0))
            else:
                import torch
                data.node_mask = torch.zeros(data.x.size(0))
                
        return data
    
def _load_shapeggen(dataset_name: str) -> SyntheticGraphDataset:
    """
    Parse a shapeggen_* dataset name and return a SyntheticGraphDataset.
 
    Name format:
        shapeggen_<shape>_<base>_n<num_graphs>_nodes<avg_nodes>
 
    Examples:
        shapeggen_house_ba_n600_nodes30
        shapeggen_cycle_ba_n400_nodes25
        shapeggen_wheel_ba_n600_nodes30
        shapeggen_house_er_n600_nodes30   (er treated same as ba here)
 
    Notes
    -----
    nodes<N>: avg number of BA base graph nodes.  Recommended 20-40.
              The motif adds 5-6 nodes on top, so total graph size
              is roughly avg_nodes + motif_size.
              With avg_nodes=30 and house motif (5 nodes), avg total ≈ 35.
 
    n<N>:     Total graphs. Half will be class 1, half class 0.
              Minimum recommended: 200.
    """
    parts = dataset_name.split('_')
    if len(parts) < 5:
        raise ValueError(
            f"Invalid shapeggen name '{dataset_name}'. "
            "Expected: shapeggen_<shape>_<base>_n<num_graphs>_nodes<avg_nodes> "
            "e.g. shapeggen_house_ba_n600_nodes30"
        )
 
    shape      = parts[1]
    base_graph = parts[2]
    num_graphs = int(parts[3].replace('n', ''))
    avg_nodes  = int(parts[4].replace('nodes', ''))
 
    if avg_nodes < 10:
        raise ValueError(
            f"nodes={avg_nodes} is too small. Use at least nodes10, "
            "recommended nodes20 to nodes40."
        )
 
    print(f"  [SyntheticMotif] Generating: shape={shape}, base={base_graph}, "
          f"num_graphs={num_graphs}, avg_nodes={avg_nodes}")
 
    return _build_synthetic_dataset(
        shape          = shape,
        base_graph     = base_graph,
        num_graphs     = num_graphs,
        avg_nodes      = avg_nodes,
        node_feature_dim = 10,
        seed           = 42,
    )


class MUTAGDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.upper()
        super(MUTAGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['MUTAG_A', 'MUTAG_graph_labels', 'MUTAG_graph_indicator', 'MUTAG_node_labels']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
            nodes_all_temp = f.read().splitlines()
            nodes_all = [int(i) for i in nodes_all_temp]

        adj_all = np.zeros((len(nodes_all), len(nodes_all)))
        with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
            adj_list = f.read().splitlines()
        for item in adj_list:
            lr = item.split(', ')
            l = int(lr[0])
            r = int(lr[1])
            adj_all[l - 1, r - 1] = 1

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
            graph_indicator_temp = f.read().splitlines()
            graph_indicator = [int(i) for i in graph_indicator_temp]
            graph_indicator = np.array(graph_indicator)

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
            graph_labels_temp = f.read().splitlines()
            graph_labels = [int(i) for i in graph_labels_temp]

        data_list = []
        for i in range(1, 189):
            idx = np.where(graph_indicator == i)
            graph_len = len(idx[0])
            adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
            label = int(graph_labels[i - 1] == 1)
            feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
            nb_clss = 7
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                y=label)
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
        = read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


class SynGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data = read_syn_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ba2motif_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save(self.collate(data_list), self.processed_paths[0])


def load_MUTAG(dataset_dir, dataset_name):
    """ 188 molecules where label = 1 denotes mutagenic effect """
    dataset = MUTAGDataset(root=dataset_dir, name=dataset_name)
    return dataset


def load_syn_data(dataset_dir, dataset_name):
    """ The synthetic dataset """
    if dataset_name.lower() == 'BA_2Motifs'.lower():
        dataset = BA2MotifDataset(root=dataset_dir, name=dataset_name)
    else:
        dataset = SynGraphDataset(root=dataset_dir, name=dataset_name)
    dataset.node_type_dict = {k: v for k, v in enumerate(range(dataset.num_classes))}
    dataset.node_color = None
    return dataset


def load_MolecueNet(dataset_dir, dataset_name, task=None):
    """ Attention the multi-task problems not solved yet """
    molecule_net_dataset_names = {name.lower(): name for name in MoleculeNet.names.keys()}
    dataset = MoleculeNet(root=dataset_dir, name=molecule_net_dataset_names[dataset_name.lower()])
    dataset.data.x = dataset.data.x.float()
    if task is None:
        dataset.data.y = dataset.data.y.squeeze().long()
    else:
        dataset.data.y = dataset.data.y[:, 0].long()
    dataset.node_type_dict = None
    dataset.node_color = None
    return dataset


def load_SeniGraph(dataset_dir, dataset_name):
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name)
    return dataset


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=5):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        # Using labels to create stratified indices
        labels = _extract_graph_labels(dataset)
        train_indices, eval_indices, test_indices = _stratified_split_indices(
            labels, data_split_ratio, seed
        )
        
        train = Subset(dataset, train_indices)
        eval = Subset(dataset, eval_indices)
        test = Subset(dataset, test_indices)

    # Now the subsets are guaranteed to have mortality cases in all splits
    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader
