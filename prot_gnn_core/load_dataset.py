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
    def __init__(self, root, name, k=10,
                 csv_filename='merged_ed_no_icd_sample_20k.csv',
                 transform=None, pre_transform=None):
        self.name = name
        self.k = k
        self.csv_filename = csv_filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        # use csv stem + k so different CSVs/k values don't collide
        csv_stem = self.csv_filename.replace('.csv', '')
        return osp.join(self.root, f'processed_patsim_{csv_stem}_k{self.k}')

    @property
    def raw_file_names(self):
        return [self.csv_filename]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        from sklearn.neighbors import NearestNeighbors

        df = pd.read_csv(osp.join(self.raw_dir, self.csv_filename))
        df = df.drop(columns=['subject_id'], errors='ignore')

        y_vals = df['disposition'].values.astype(int)
        df = df.drop(columns=['disposition'])

        feature_cols = df.columns.tolist()

        # z-score normalize continuous columns; leave binary (0/1) columns as-is
        for col in feature_cols:
            if df[col].nunique() > 2:
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - df[col].mean()) / std
        df = df.fillna(0)

        X = df.values.astype(np.float32)
        print(f"  Building k={self.k} nearest-neighbour graph for {len(X)} patients "
              f"({len(feature_cols)} features)...")

        nn_model = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine', n_jobs=-1)
        nn_model.fit(X)
        _, indices = nn_model.kneighbors(X)   # shape [n, k+1]; col 0 = self

        # fixed star edge list (reused for every graph)
        src = [0] * self.k + list(range(1, self.k + 1))
        dst = list(range(1, self.k + 1)) + [0] * self.k
        edge_index_template = torch.tensor([src, dst], dtype=torch.long)

        data_list = []
        for i in range(len(X)):
            neighbour_idx = indices[i][1:]            # skip self
            node_idx = [i] + list(neighbour_idx)      # center + k neighbours

            x = torch.tensor(X[node_idx], dtype=torch.float)
            y = torch.tensor([y_vals[i]], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index_template.clone(), y=y))

        print(f"  Done. {len(data_list)} graphs, "
              f"{self.k + 1} nodes/graph, input_dim={len(feature_cols)}")
        torch.save(self.collate(data_list), self.processed_paths[0])


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
    elif dataset_name.lower().startswith('mimic_patient_sim'):
        # name format:  mimic_patient_sim[_full][_k<N>]
        # e.g.  mimic_patient_sim          → sample 20k, k=10
        #       mimic_patient_sim_k20      → sample 20k, k=20
        #       mimic_patient_sim_full_k20 → full dataset, k=20
        name_l = dataset_name.lower()
        full   = 'full' in name_l
        csv    = 'merged_ed_no_icd.csv' if full else 'merged_ed_no_icd_sample_20k.csv'
        k      = 10
        for part in name_l.split('_'):
            if part.startswith('k') and part[1:].isdigit():
                k = int(part[1:])
        return PatientSimilarityGraphDataset(root=dataset_dir, name=dataset_name,
                                             k=k, csv_filename=csv)
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
