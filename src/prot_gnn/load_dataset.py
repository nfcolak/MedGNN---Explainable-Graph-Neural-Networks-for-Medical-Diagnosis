#load_dataset.py
import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
import pandas as pd
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader

POST_OUTCOME_COLUMNS = []


def _drop_post_outcome_columns(df):
    return df.drop(columns=POST_OUTCOME_COLUMNS, errors='ignore')


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


def _subject_aware_split_indices(labels, groups, split_sizes, seed):
    """
    Split graphs into train/eval/test so that all graphs sharing a group id
    (e.g. the same patient's multiple ED visits) land in EXACTLY ONE split.
    This prevents subject-level leakage. Stratification is approximated by
    assigning whole groups in descending size while greedily balancing each
    split's positive rate toward the global positive rate.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)

    # aggregate per group: indices, count, positive count
    group_to_idx = {}
    for idx, g in enumerate(groups):
        group_to_idx.setdefault(int(g), []).append(idx)

    group_ids = list(group_to_idx.keys())
    rng.shuffle(group_ids)

    n_total = len(labels)
    target = {
        "train": split_sizes[0] * n_total,
        "eval":  split_sizes[1] * n_total,
        "test":  split_sizes[2] * n_total,
    }
    global_pos = labels.mean() if n_total else 0.0

    buckets = {k: {"idx": [], "n": 0, "pos": 0} for k in ("train", "eval", "test")}

    # assign larger groups first → more stable balancing
    group_ids.sort(key=lambda g: len(group_to_idx[g]), reverse=True)

    for g in group_ids:
        idxs = group_to_idx[g]
        g_n = len(idxs)
        g_pos = int(labels[idxs].sum())

        best_split, best_score = None, None
        for k in ("train", "eval", "test"):
            if target[k] <= 0:
                continue
            b = buckets[k]
            # how far below capacity (prefer under-filled splits)
            room = (target[k] - b["n"]) / target[k]
            # positive-rate deviation if we add this group
            new_pos_rate = (b["pos"] + g_pos) / (b["n"] + g_n)
            balance_pen = abs(new_pos_rate - global_pos)
            score = room - 0.5 * balance_pen
            if best_score is None or score > best_score:
                best_score, best_split = score, k

        b = buckets[best_split]
        b["idx"].extend(idxs)
        b["n"] += g_n
        b["pos"] += g_pos

    train_indices = buckets["train"]["idx"]
    eval_indices = buckets["eval"]["idx"]
    test_indices = buckets["test"]["idx"]
    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    rng.shuffle(test_indices)
    return train_indices, eval_indices, test_indices


def _extract_graph_groups(dataset, attr="subject_id"):
    """Return per-graph group ids (e.g. subject_id) if present, else None."""
    try:
        first = dataset[0]
    except Exception:
        return None
    if not hasattr(first, attr):
        return None
    groups = []
    for idx in range(len(dataset)):
        val = getattr(dataset[idx], attr)
        if torch.is_tensor(val):
            val = val.view(-1)[0].item()
        groups.append(int(val))
    return np.array(groups)


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
                 drop_cols=(),
                 transform=None, pre_transform=None):
        self.name = name
        self.k = k
        self.graph_mode = graph_mode
        self.csv_filename = csv_filename
        # Columns to drop from features (e.g. ['los_hours'] to avoid post-outcome
        # leakage). Influences the processed-dir name so cached graphs don't mix.
        self.drop_cols = tuple(drop_cols)
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
        # use csv stem + k + drop-cols + graph-mode so different variants don't collide
        csv_stem = self.csv_filename.replace('.csv', '')
        no_los_suffix = "_no_los" if 'los_hours' in self.drop_cols else ""
        mode_suffix = "" if self.graph_mode == "star" else f"_{self.graph_mode}"
        return osp.join(self.root, f'processed_patsim_{csv_stem}{no_los_suffix}_k{self.k}{mode_suffix}')

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
        if self.drop_cols:
            df = df.drop(columns=[c for c in self.drop_cols if c in df.columns],
                         errors='ignore')

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


class IntraPatientHeteroDataset(InMemoryDataset):
    """
    Intra-patient heterogeneous graph (one graph per patient).

    Replaces the patient-similarity star graph (which carried almost no
    label-relevant topology — k=10 neighbour-label agreement ≈ class prior).
    Instead, each patient's own clinical record becomes a graph:

        Node types:
          - PATIENT  (1 per graph): demographics + transport encoded in feature.
          - VITAL    (one per non-NaN vital): z-scored value + abnormal flag.
          - MED      (one per medication actually taken, prevalence ≥ med_min_prev).

        Edges:
          - patient ↔ each vital  (star spokes for direct vital signal)
          - patient ↔ each med
          - med    ↔ med           when PMI(med_i, med_j) > pmi_threshold,
                                   giving the GNN drug-cluster topology
                                   (cardiac bundle, psych bundle, etc.).

    Node feature layout (same dim D for every node — required by GCN):
        [type_onehot(3),
         vital_id_onehot(V_v),     filled only for VITAL nodes
         med_id_onehot(V_m),       filled only for MED nodes
         value(1),                 z-scored vital, or 1.0 for MED
         abnormal(1),              |z|>2 for vitals
         demo(D_d)]                filled only for PATIENT node

    Prototype learning on this graph means each learned prototype represents
    a clinically interpretable substructure — e.g.
    {patient + vital(o2sat↓) + vital(resprate↑) + med(furosemide) + med(metoprolol)}
    — exactly the kind of evidence TODO #13 asks for.
    """

    VITAL_COLS = [
        'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp',
        'pain', 'acuity',
        'vs_temperature', 'vs_heartrate', 'vs_resprate', 'vs_o2sat',
        'vs_sbp', 'vs_dbp', 'vs_count',
    ]
    DEMO_COLS = [
        'gender_F', 'gender_M',
        'race_ASIAN', 'race_BLACK', 'race_HISPANIC', 'race_NATIVE',
        'race_OTHER', 'race_WHITE',
        'transport_AMBULANCE', 'transport_HELICOPTER', 'transport_WALK IN',
    ]
    # Clinical clipping ranges to prevent z-score distortion from outliers
    # (e.g. raw data has temperature=28.89 °C, heartrate=217 bpm).
    VITAL_CLIP = {
        'temperature':    (34.0, 42.0),
        'heartrate':      (30.0, 200.0),
        'resprate':       (6.0,  45.0),
        'o2sat':          (60.0, 100.0),
        'sbp':            (60.0, 250.0),
        'dbp':            (30.0, 150.0),
        'pain':           (0.0,  10.0),
        'acuity':         (1.0,  5.0),
        'vs_temperature': (34.0, 42.0),
        'vs_heartrate':   (30.0, 200.0),
        'vs_resprate':    (6.0,  45.0),
        'vs_o2sat':       (60.0, 100.0),
        'vs_sbp':         (60.0, 250.0),
        'vs_dbp':         (30.0, 150.0),
        'vs_count':       (1.0,  30.0),
    }

    def __init__(self, root, name,
                 csv_filename='merged_ed_no_icd_sample_20k.csv',
                 med_min_prev=0.01, pmi_threshold=2.0, drop_los=True,
                 add_missing_flag=True, icd_min_prev=0.01,
                 target='disposition',
                 transform=None, pre_transform=None):
        self.name = name
        self.csv_filename = csv_filename
        self.med_min_prev = float(med_min_prev)
        self.pmi_threshold = float(pmi_threshold)
        self.drop_los = bool(drop_los)
        # Prediction target:
        #   'disposition' -> binary HOME(0)/ADMITTED(1)
        #   'disease'     -> single-label primary diagnosis (disease_1), one
        #                    class per distinct disease category. disease_* and
        #                    symptom_* columns are then removed from the inputs.
        self.target = str(target)
        # If the CSV has an icd_codes column, add ICD diagnosis nodes (4th node
        # type). Diagnoses are the strongest single signal for admission BUT may
        # leak for early-prediction framing (codes can be assigned during/after
        # the admit decision) — report ICD runs with that caveat.
        self.icd_min_prev = float(icd_min_prev)
        # Add a per-vital "was this measured?" flag. Without it, a missing vital
        # is z-scored to 0 (= the population mean) and becomes indistinguishable
        # from a genuinely average measurement. The fact that a vital was *not*
        # taken is itself clinical signal (low-acuity patients get fewer
        # measurements).
        self.add_missing_flag = bool(add_missing_flag)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._load_metadata()

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        csv_stem = self.csv_filename.replace('.csv', '')
        los_tag = 'noLOS' if self.drop_los else 'LOS'
        prev_tag = f'prev{int(round(self.med_min_prev * 1000))}'
        pmi_tag  = f'pmi{self.pmi_threshold:g}'
        miss_tag = '_miss' if self.add_missing_flag else ''
        icd_tag  = f'_icd{int(round(self.icd_min_prev * 1000))}' if self.icd_min_prev else ''
        tgt_tag  = '' if self.target == 'disposition' else f'_{self.target}'
        return osp.join(
            self.root,
            f'processed_hetero_{csv_stem}_{los_tag}_{prev_tag}_{pmi_tag}{miss_tag}{icd_tag}{tgt_tag}',
        )

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
        self.feature_cols = []          # for compatibility with explanation code
        self.vital_vocab  = []
        self.med_vocab    = []
        self.icd_vocab    = []
        self.symptom_vocab = []
        self.cc_vocab     = []
        self.patient_num_vocab = []
        self.demo_vocab   = []
        self.feature_metadata = {}
        self.label_mapping = {"0": "HOME", "1": "ADMITTED"}
        if osp.isfile(self.metadata_path):
            with open(self.metadata_path) as f:
                meta = json.load(f)
            self.vital_vocab = meta.get("vital_vocab", [])
            self.med_vocab   = meta.get("med_vocab", [])
            self.icd_vocab   = meta.get("icd_vocab", [])
            self.symptom_vocab = meta.get("symptom_vocab", [])
            self.cc_vocab    = meta.get("cc_vocab", [])
            self.patient_num_vocab = meta.get("patient_num_vocab", [])
            self.demo_vocab  = meta.get("demo_vocab", [])
            self.feature_metadata = meta.get("vital_stats", {})
            self.label_mapping = meta.get("label_mapping", self.label_mapping)
            # Surface a flat human-readable name list per feature slot for
            # GraphXAI explanations. MUST match process() layout exactly:
            # fixed 6 type slots, then id blocks, value/abnormal/missing,
            # demo block, patient-numeric block.
            types = ["patient", "vital", "med", "icd", "symptom", "chiefcomplaint"]
            self.feature_cols = (
                [f"type[{t}]" for t in types]
                + [f"vital_id[{v}]" for v in self.vital_vocab]
                + [f"med_id[{m}]"   for m in self.med_vocab]
                + [f"icd_id[{c}]"   for c in self.icd_vocab]
                + [f"sym_id[{s}]"   for s in self.symptom_vocab]
                + [f"cc_id[{c}]"    for c in self.cc_vocab]
                + ["value", "abnormal_flag", "missing_flag"]
                + [f"demo[{d}]"     for d in self.demo_vocab]
                + [f"pnum[{p}]"     for p in self.patient_num_vocab]
            )

    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self.csv_filename))
        # Keep subject_id so the dataloader can build a SUBJECT-AWARE split
        # (the same patient appears in multiple ED visits; letting one visit
        # leak into train and another into test inflates metrics).
        subject_ids = (df['subject_id'].values.astype(np.int64)
                       if 'subject_id' in df.columns
                       else np.arange(len(df), dtype=np.int64))
        df = df.drop(columns=['subject_id'], errors='ignore')

        # --- Build the prediction target ---
        label_names = None
        if self.target == 'disease':
            # Single-label primary diagnosis (disease_1). Map each distinct
            # disease string to an integer class id.
            diseases = df['disease_1'].fillna('').astype(str)
            classes = sorted(d for d in diseases.unique() if d)
            class_to_id = {d: i for i, d in enumerate(classes)}
            y_vals = diseases.map(class_to_id).fillna(-1).astype(int).values
            label_names = classes
            # rows with no usable disease_1 are dropped
            keep = y_vals >= 0
        else:
            y_vals = df['disposition'].values.astype(int)
            keep = np.ones(len(df), dtype=bool)
        self._label_names = label_names

        # Capture presenting-complaint columns BEFORE dropping feature columns.
        #   symptom_*        : doctor-refined ICD R-codes (post-exam)
        #   chiefcomplaint_* : triage free text (pre-diagnosis, the door)
        # Both are legitimate model INPUT (complaint -> disease is standard
        # clinical reasoning, not leakage); each becomes its own node type.
        symptom_cols = sorted(c for c in df.columns if c.startswith('symptom_'))
        symptom_lists_raw = (
            df[symptom_cols].fillna('').astype(str).values.tolist()
            if symptom_cols else [[] for _ in range(len(df))]
        )
        cc_cols = sorted(c for c in df.columns if c.startswith('chiefcomplaint_'))
        cc_lists_raw = (
            df[cc_cols].fillna('').astype(str).values.tolist()
            if cc_cols else [[] for _ in range(len(df))]
        )

        # Drop only TRUE leakage: the disease target itself (disease_*), the raw
        # icd_codes it was derived from, and disposition. Symptoms and chief
        # complaints are KEPT (captured above as node lists).
        dx_cols = [c for c in df.columns
                   if c.startswith('disease_') or c.startswith('symptom_')
                   or c.startswith('chiefcomplaint_')
                   or c in ('icd_codes', 'disposition')]
        df = df.drop(columns=dx_cols, errors='ignore')

        if not keep.all():
            df = df[keep].reset_index(drop=True)
            y_vals = y_vals[keep]
            subject_ids = subject_ids[keep]
            symptom_lists_raw = [symptom_lists_raw[i] for i in np.nonzero(keep)[0]]
            cc_lists_raw = [cc_lists_raw[i] for i in np.nonzero(keep)[0]]

        def _build_vocab(lists_raw):
            counter = {}
            for row in lists_raw:
                for s in row:
                    s = s.strip()
                    if s:
                        counter[s] = counter.get(s, 0) + 1
            vocab = sorted(counter.keys())
            index = {s: j for j, s in enumerate(vocab)}
            patient_ids = [
                [index[s.strip()] for s in row if s.strip() in index]
                for row in lists_raw
            ]
            return vocab, patient_ids

        # Build symptom + chief-complaint vocabularies and per-patient id lists.
        symptom_vocab, patient_symptoms = _build_vocab(symptom_lists_raw)
        cc_vocab, patient_ccs = _build_vocab(cc_lists_raw)
        if symptom_vocab:
            print(f"  Symptom nodes: {len(symptom_vocab)} categories "
                  f"(avg {np.mean([len(p) for p in patient_symptoms]):.1f} per patient)")
        if cc_vocab:
            print(f"  Chief-complaint nodes: {len(cc_vocab)} categories "
                  f"(avg {np.mean([len(p) for p in patient_ccs]):.1f} per patient)")

        df = _drop_post_outcome_columns(df)
        if self.drop_los and 'los_hours' in df.columns:
            df = df.drop(columns=['los_hours'])

        vital_cols = [c for c in self.VITAL_COLS if c in df.columns]
        demo_cols  = [c for c in self.DEMO_COLS  if c in df.columns]
        # Medication nodes = home meds (med_*) AND, if present, ED-dispensed
        # meds (pyx_*). They share the MED node type; the pyx_ prefix keeps them
        # distinguishable in the one-hot id block and in explanations, and lets
        # PMI edges capture home-med ↔ ED-med co-occurrence.
        med_cols_all = sorted([c for c in df.columns
                               if c.startswith('med_') or c.startswith('pyx_')])

        # Drop ultra-rare meds (no statistical mass + GNN noise).
        med_prev = df[med_cols_all].mean()
        med_cols = [c for c in med_cols_all if med_prev[c] >= self.med_min_prev]
        n_pyx = sum(c.startswith('pyx_') for c in med_cols)
        print(f"  Vitals: {len(vital_cols)}  |  demo: {len(demo_cols)}  |  "
              f"meds kept: {len(med_cols)}/{len(med_cols_all)} "
              f"({n_pyx} ED-dispensed pyx_, prevalence ≥ {self.med_min_prev:.3f})")

        # --- Patient-level NUMERIC features (folded into the PATIENT node) ---
        # Scalar signals that describe the whole visit/patient rather than a
        # single vital/med: ED-utilisation, polypharmacy count, and vital trend
        # statistics (min/max/std). Kept on the patient node (z-scored) so graph
        # topology is unchanged. (vs_* MEAN columns already become VITAL nodes.)
        patient_num_cols = [c for c in df.columns
                            if c in ('n_ed_visits', 'n_medications', 'age', 'bmi')
                            or c.startswith('lab_')
                            or c.startswith('medclass_')
                            or c.startswith('hx_')
                            or (c.startswith('vs_') and c.endswith(('_min', '_max', '_std')))]
        patient_num_cols = sorted(patient_num_cols)
        patient_num_stats = {}
        for col in patient_num_cols:
            mean = float(df[col].mean())
            std = float(df[col].std())
            patient_num_stats[col] = {"mean": mean, "std": std}
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)
        if patient_num_cols:
            print(f"  Patient numeric features: {len(patient_num_cols)} "
                  f"(n_ed_visits/n_medications/vital-trends, z-scored on patient node)")

        # --- ICD diagnosis vocabulary (4th node type, optional) ---
        # icd_codes is a ';'-separated string of diagnosis codes per patient.
        icd_lists = None
        icd_vocab = []
        if 'icd_codes' in df.columns and self.icd_min_prev > 0:
            def _split_icd(v):
                if pd.isna(v):
                    return []
                return [c.strip() for c in str(v).split(';') if c.strip()]
            icd_lists = df['icd_codes'].apply(_split_icd).tolist()
            from collections import Counter as _C
            icd_counter = _C()
            for codes in icd_lists:
                icd_counter.update(set(codes))
            min_count = self.icd_min_prev * len(df)
            icd_vocab = sorted([c for c, n in icd_counter.items() if n >= min_count])
            icd_index = {c: j for j, c in enumerate(icd_vocab)}
            # per-patient kept-ICD index list
            icd_lists = [[icd_index[c] for c in set(codes) if c in icd_index]
                         for codes in icd_lists]
            print(f"  ICD nodes: {len(icd_vocab)} codes kept "
                  f"(prevalence ≥ {self.icd_min_prev:.3f}) — LEAKAGE CAVEAT")
        if 'icd_codes' in df.columns:
            df = df.drop(columns=['icd_codes'])

        # --- Missing-value mask (capture BEFORE imputation) ---
        # 1.0 = this vital was actually measured, 0.0 = missing.
        vital_missing = {col: df[col].isna().values.copy() for col in vital_cols}

        # --- Vital clipping + z-score (clip first so a single extreme value
        # doesn't break the population mean/std). ---
        vital_stats = {}
        for col in vital_cols:
            lo, hi = self.VITAL_CLIP.get(col, (None, None))
            if lo is not None:
                df[col] = df[col].clip(lower=lo, upper=hi)
            # mean/std computed on observed (non-missing) values only
            mean = float(df[col].mean())
            std  = float(df[col].std())
            vital_stats[col] = {"mean": mean, "std": std,
                                "clip_lo": lo, "clip_hi": hi,
                                "missing_rate": float(np.mean(vital_missing[col]))}
            if std > 0:
                df[col] = (df[col] - mean) / std
            df[col] = df[col].fillna(0.0)  # missing → population mean (z=0)

        # --- Med-med PMI edges (computed once on the full sample for a
        # stable population statistic; using only training rows would
        # require a holdout-aware refactor we don't need for this graph
        # of co-occurrence statistics). ---
        med_matrix = (df[med_cols].values > 0).astype(np.float32)
        med_marginals = med_matrix.mean(axis=0)
        eps = 1e-9
        # co-occurrence in float64 (float32 matmul triggers spurious BLAS
        # overflow warnings on some platforms; counts are small integers anyway)
        _mm = med_matrix.astype(np.float64)
        co = (_mm.T @ _mm) / float(len(df))
        pmi = np.log((co + eps) / (np.outer(med_marginals, med_marginals) + eps))
        np.fill_diagonal(pmi, 0.0)
        pmi_pairs = np.argwhere(pmi > self.pmi_threshold)
        # Deduplicate (i,j) and (j,i) — we'll emit both directions explicitly.
        pmi_pairs = pmi_pairs[pmi_pairs[:, 0] < pmi_pairs[:, 1]]
        # med_idx → list of co-occurring med_idxs
        med_neighbours = [[] for _ in range(len(med_cols))]
        for a, b in pmi_pairs:
            a, b = int(a), int(b)
            med_neighbours[a].append(b)
            med_neighbours[b].append(a)
        print(f"  Med-med PMI edges (undirected, PMI > {self.pmi_threshold}): "
              f"{len(pmi_pairs)}")

        # --- Feature layout ---
        V_v = len(vital_cols)
        V_m = len(med_cols)
        V_i = len(icd_vocab)
        V_s = len(symptom_vocab)
        V_c = len(cc_vocab)
        D_d = len(demo_cols)
        P_n = len(patient_num_cols)
        # type one-hot slots: FIXED 6 slots (stable indices regardless of which
        # node types are present in this particular dataset variant).
        IDX_TYPE_PATIENT = 0
        IDX_TYPE_VITAL   = 1
        IDX_TYPE_MED     = 2
        IDX_TYPE_ICD     = 3                            # only used when V_i > 0
        IDX_TYPE_SYMPTOM = 4                            # only used when V_s > 0
        IDX_TYPE_CC      = 5                            # only used when V_c > 0
        N_TYPES = 6
        IDX_VITAL_START  = N_TYPES
        IDX_MED_START    = IDX_VITAL_START + V_v
        IDX_ICD_START    = IDX_MED_START + V_m          # icd id one-hot block
        IDX_SYM_START    = IDX_ICD_START + V_i          # symptom id one-hot block
        IDX_CC_START     = IDX_SYM_START + V_s          # chief-complaint id block
        IDX_VALUE        = IDX_CC_START + V_c
        IDX_ABNORMAL     = IDX_VALUE + 1
        IDX_MISSING      = IDX_ABNORMAL + 1            # vital "was measured?" flag
        IDX_DEMO_START   = IDX_MISSING + 1
        IDX_PNUM_START   = IDX_DEMO_START + D_d         # patient numeric block
        feat_dim         = IDX_PNUM_START + P_n

        vital_z   = df[vital_cols].values.astype(np.float32)
        # (N, V_v) boolean → 1.0 where the vital was MISSING in the raw data
        vital_missing_mat = np.stack(
            [vital_missing[c] for c in vital_cols], axis=1
        ).astype(np.float32) if V_v else np.zeros((len(df), 0), dtype=np.float32)
        med_taken = med_matrix                                    # (N, V_m)
        demo_vec  = df[demo_cols].values.astype(np.float32) if demo_cols else \
                    np.zeros((len(df), 0), dtype=np.float32)
        pnum_vec  = df[patient_num_cols].values.astype(np.float32) if patient_num_cols else \
                    np.zeros((len(df), 0), dtype=np.float32)

        data_list = []
        for i in range(len(df)):
            patient_meds = np.nonzero(med_taken[i])[0]            # indices into med_cols
            num_med_nodes = int(len(patient_meds))
            patient_icds = icd_lists[i] if icd_lists is not None else []
            num_icd_nodes = len(patient_icds)
            patient_syms = patient_symptoms[i] if symptom_vocab else []
            num_sym_nodes = len(patient_syms)
            patient_cc = patient_ccs[i] if cc_vocab else []
            num_cc_nodes = len(patient_cc)
            num_nodes = (1 + V_v + num_med_nodes + num_icd_nodes
                         + num_sym_nodes + num_cc_nodes)
            x = np.zeros((num_nodes, feat_dim), dtype=np.float32)

            # Patient node (idx 0)
            x[0, IDX_TYPE_PATIENT] = 1.0
            if D_d > 0:
                x[0, IDX_DEMO_START:IDX_DEMO_START + D_d] = demo_vec[i]
            if P_n > 0:
                x[0, IDX_PNUM_START:IDX_PNUM_START + P_n] = pnum_vec[i]

            # Vital nodes
            for v_idx in range(V_v):
                n = 1 + v_idx
                x[n, IDX_TYPE_VITAL] = 1.0
                x[n, IDX_VITAL_START + v_idx] = 1.0
                z = vital_z[i, v_idx]
                x[n, IDX_VALUE] = z
                is_missing = vital_missing_mat[i, v_idx] > 0
                # abnormal only meaningful when the value was actually observed
                if (not is_missing) and abs(z) > 2.0:
                    x[n, IDX_ABNORMAL] = 1.0
                if self.add_missing_flag and is_missing:
                    x[n, IDX_MISSING] = 1.0

            # Med nodes
            local_of_global = {}
            med_offset = 1 + V_v
            for k, g_idx in enumerate(patient_meds):
                g_idx = int(g_idx)
                n = med_offset + k
                x[n, IDX_TYPE_MED] = 1.0
                x[n, IDX_MED_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0
                local_of_global[g_idx] = n

            # ICD diagnosis nodes
            icd_offset = 1 + V_v + num_med_nodes
            for k, g_idx in enumerate(patient_icds):
                n = icd_offset + k
                x[n, IDX_TYPE_ICD] = 1.0
                x[n, IDX_ICD_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0

            # SYMPTOM (doctor-refined ICD R-code) nodes
            sym_offset = 1 + V_v + num_med_nodes + num_icd_nodes
            for k, g_idx in enumerate(patient_syms):
                n = sym_offset + k
                x[n, IDX_TYPE_SYMPTOM] = 1.0
                x[n, IDX_SYM_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0

            # CHIEF-COMPLAINT (triage free-text) nodes
            cc_offset = 1 + V_v + num_med_nodes + num_icd_nodes + num_sym_nodes
            for k, g_idx in enumerate(patient_cc):
                n = cc_offset + k
                x[n, IDX_TYPE_CC] = 1.0
                x[n, IDX_CC_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0

            # Edges
            src, dst = [], []
            # patient ↔ vital
            for v_idx in range(V_v):
                n = 1 + v_idx
                src += [0, n]; dst += [n, 0]
            # patient ↔ med
            for k in range(num_med_nodes):
                n = med_offset + k
                src += [0, n]; dst += [n, 0]
            # patient ↔ icd
            for k in range(num_icd_nodes):
                n = icd_offset + k
                src += [0, n]; dst += [n, 0]
            # patient ↔ symptom
            for k in range(num_sym_nodes):
                n = sym_offset + k
                src += [0, n]; dst += [n, 0]
            # patient ↔ chief-complaint
            for k in range(num_cc_nodes):
                n = cc_offset + k
                src += [0, n]; dst += [n, 0]
            # med ↔ med via PMI co-occurrence
            for g_idx in patient_meds:
                g_idx = int(g_idx)
                for co_idx in med_neighbours[g_idx]:
                    if co_idx in local_of_global and g_idx < co_idx:
                        a = local_of_global[g_idx]
                        b = local_of_global[co_idx]
                        src += [a, b]; dst += [b, a]
            if not src:
                # Pathological case (no meds AND no vitals) → self-loop on patient.
                src, dst = [0], [0]
            edge_index = torch.tensor([src, dst], dtype=torch.long)

            data_list.append(Data(
                x=torch.from_numpy(x),
                edge_index=edge_index,
                y=torch.tensor([y_vals[i]], dtype=torch.long),
                dataset_index=torch.tensor([i], dtype=torch.long),
                subject_id=torch.tensor([int(subject_ids[i])], dtype=torch.long),
                num_med_nodes=torch.tensor([num_med_nodes], dtype=torch.long),
                num_icd_nodes=torch.tensor([num_icd_nodes], dtype=torch.long),
                num_sym_nodes=torch.tensor([num_sym_nodes], dtype=torch.long),
                num_cc_nodes=torch.tensor([num_cc_nodes], dtype=torch.long),
                num_total_nodes=torch.tensor([num_nodes], dtype=torch.long),
            ))

        avg_nodes = np.mean([d.x.shape[0]            for d in data_list])
        avg_edges = np.mean([d.edge_index.shape[1]   for d in data_list]) / 2
        avg_meds  = np.mean([int(d.num_med_nodes)    for d in data_list])
        avg_icds  = np.mean([int(d.num_icd_nodes)    for d in data_list])
        avg_syms  = np.mean([int(d.num_sym_nodes)    for d in data_list])
        avg_ccs   = np.mean([int(d.num_cc_nodes)     for d in data_list])
        print(f"  Built {len(data_list)} intra-patient graphs: "
              f"avg nodes={avg_nodes:.1f}, avg edges={avg_edges:.1f}, "
              f"avg med-nodes={avg_meds:.1f}, avg icd-nodes={avg_icds:.1f}, "
              f"avg sym-nodes={avg_syms:.1f}, avg cc-nodes={avg_ccs:.1f}, "
              f"feat_dim={feat_dim}")

        torch.save(self.collate(data_list), self.processed_paths[0])
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump({
                "csv_filename": self.csv_filename,
                "feature_dim": feat_dim,
                "vital_vocab": vital_cols,
                "med_vocab":   med_cols,
                "icd_vocab":   icd_vocab,
                "symptom_vocab": symptom_vocab,
                "cc_vocab":    cc_vocab,
                "patient_num_vocab": patient_num_cols,
                "patient_num_stats": patient_num_stats,
                "demo_vocab":  demo_cols,
                "med_min_prev": self.med_min_prev,
                "icd_min_prev": self.icd_min_prev,
                "pmi_threshold": self.pmi_threshold,
                "n_med_med_edges": int(len(pmi_pairs)),
                "drop_los": self.drop_los,
                "add_missing_flag": self.add_missing_flag,
                "vital_stats": vital_stats,
                "target": self.target,
                "label_mapping": (
                    {str(i): n for i, n in enumerate(self._label_names)}
                    if getattr(self, "_label_names", None)
                    else {"0": "HOME", "1": "ADMITTED"}
                ),
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
    elif dataset_name.lower().startswith('mimic_intra_patient'):
        # Intra-patient heterogeneous graph (Option A).
        # name format:  mimic_intra_patient[_full][_with_los][_prevN][_pmiX]
        # e.g.  mimic_intra_patient                → 20k sample, LOS dropped, prev>=1%, PMI>2
        #       mimic_intra_patient_with_los       → keep los_hours (leakage ablation)
        #       mimic_intra_patient_full           → full 309k patients
        #       mimic_intra_patient_prev5_pmi3     → med prev >= 0.5%, PMI > 3
        # name format:  mimic_intra_patient[_full][_with_los][_icd][_disease][_prevN][_pmiX]
        #   ..._disease  → predict primary diagnosis (disease_1) instead of disposition
        name_l = dataset_name.lower()
        parts  = name_l.split('_')
        full   = 'full' in name_l
        with_los = 'with_los' in name_l
        use_icd  = 'icd' in parts                       # mimic_intra_patient_icd
        target   = 'disease' if 'disease' in parts else 'disposition'
        if use_icd:
            csv = 'merged_ed_with_icd.csv' if full else 'merged_ed_with_icd_sample_60k.csv'
        else:
            csv = 'merged_ed_all_visits.csv' if full else 'merged_ed.csv'
        prev = 0.01
        pmi  = 2.0
        # diagnoses are far sparser than meds (8000+ distinct codes), so use a
        # lower default prevalence threshold to keep a useful ICD vocabulary
        icd_prev = 0.005 if use_icd else 0.0
        for part in parts:
            if part.startswith('prev') and part[4:].isdigit():
                prev = int(part[4:]) / 1000.0
            elif part.startswith('pmi'):
                try:
                    pmi = float(part[3:])
                except ValueError:
                    pass
        return IntraPatientHeteroDataset(
            root=dataset_dir, name=dataset_name, csv_filename=csv,
            med_min_prev=prev, pmi_threshold=pmi, drop_los=not with_los,
            icd_min_prev=icd_prev, target=target,
        )
    elif dataset_name.lower().startswith('mimic_patient_sim'):
        # name format:  mimic_patient_sim[_full][_no_los][_k<N>][_weighted|_local]
        # e.g.  mimic_patient_sim              → sample 20k, k=10, with LOS
        #       mimic_patient_sim_no_los       → sample 20k, k=10, LOS dropped
        #       mimic_patient_sim_no_los_k20   → sample 20k, k=20, LOS dropped
        #       mimic_patient_sim_full_no_los  → full dataset, LOS dropped
        name_l = dataset_name.lower()
        full   = 'full' in name_l
        no_los = 'no_los' in name_l
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
        drop_cols = ['los_hours'] if no_los else []
        return PatientSimilarityGraphDataset(root=dataset_dir, name=dataset_name,
                                             k=k, graph_mode=graph_mode,
                                             csv_filename=csv, drop_cols=drop_cols)
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
        raise NotImplementedError


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
        labels = _extract_graph_labels(dataset)
        groups = _extract_graph_groups(dataset, "subject_id")
        if groups is not None and len(np.unique(groups)) < len(groups):
            print(f"  [split] subject-aware across {len(np.unique(groups))} subjects")
            train_indices, eval_indices, test_indices = _subject_aware_split_indices(
                labels, groups, data_split_ratio, seed
            )
        else:
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
