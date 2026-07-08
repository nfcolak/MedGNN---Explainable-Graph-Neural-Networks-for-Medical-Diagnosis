#load_dataset.py
import os
import json
import torch
import numpy as np
import os.path as osp
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader

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
                 target='disposition', graph_structure='star',
                 transform=None, pre_transform=None):
        self.name = name
        self.csv_filename = csv_filename
        self.med_min_prev = float(med_min_prev)
        self.pmi_threshold = float(pmi_threshold)
        self.drop_los = bool(drop_los)
        # Graph structure (topology) — how the patient's concept nodes connect.
        # Validated against the shared registry so ProtGNN and GraphCare speak
        # the same names. The PATIENT hub↔concept spokes are ALWAYS present;
        # `graph_structure` controls the concept↔concept edges added on top:
        #   star     none          cooccur  PMI cross-type
        #   ontology same-class     full     cooccur ∪ ontology
        from shared.lib.graph_structures import resolve as _resolve_structure
        self.graph_structure = _resolve_structure(graph_structure, "protgnn")
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
        # Every graph type lives under the single main folder data/graphs/,
        # one subfolder per structure -> data/graphs/<structure>/protgnn/<variant>.
        from shared.lib.graph_structures import structure_dir
        return osp.join(
            str(structure_dir(self.root, self.graph_structure, "protgnn")),
            f'hetero_{csv_stem}_{los_tag}_{prev_tag}_{pmi_tag}{miss_tag}{icd_tag}{tgt_tag}',
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

        # ------------------------------------------------------------------
        # Concept↔concept edge maps, keyed by a GLOBAL concept id that spans
        # every variable concept type in a fixed block layout:
        #     [0, V_m)                 meds
        #     [V_m, V_m+V_i)           icd codes
        #     [V_m+V_i, +V_s)          symptoms
        #     [.., +V_c)               chief complaints
        # Vitals are excluded (present in every patient -> PMI ~ 0, no signal).
        # These maps are consumed per-patient in the edge builder below; which
        # of them is actually used depends on self.graph_structure.
        # ------------------------------------------------------------------
        CG_MED, CG_ICD = 0, V_m
        CG_SYM, CG_CC  = V_m + V_i, V_m + V_i + V_s
        n_concept_gids = V_m + V_i + V_s + V_c

        # Per-patient presence over the global concept space (for PMI).
        concept_present = np.zeros((len(df), n_concept_gids), dtype=np.float32)
        concept_present[:, CG_MED:CG_MED + V_m] = med_matrix
        for i in range(len(df)):
            if icd_lists is not None:
                for g in icd_lists[i]:
                    concept_present[i, CG_ICD + g] = 1.0
            if symptom_vocab:
                for g in patient_symptoms[i]:
                    concept_present[i, CG_SYM + g] = 1.0
            if cc_vocab:
                for g in patient_ccs[i]:
                    concept_present[i, CG_CC + g] = 1.0

        # cooccur map: PMI over the concept presence matrix (same statistic and
        # threshold as the med-med edges, now cross-type).
        concept_cooccur = [[] for _ in range(n_concept_gids)]
        if self.graph_structure in ('cooccur', 'full') and n_concept_gids > 1:
            Cf = concept_present.astype(np.float64)
            marg = Cf.mean(axis=0)
            co_c = (Cf.T @ Cf) / float(len(df))
            pmi_c = np.log((co_c + eps) / (np.outer(marg, marg) + eps))
            np.fill_diagonal(pmi_c, 0.0)
            pairs_c = np.argwhere(pmi_c > self.pmi_threshold)
            pairs_c = pairs_c[pairs_c[:, 0] < pairs_c[:, 1]]
            for a, b in pairs_c:
                a, b = int(a), int(b)
                concept_cooccur[a].append(b)
                concept_cooccur[b].append(a)
            print(f"  Concept-concept PMI edges (cross-type, PMI > "
                  f"{self.pmi_threshold}): {len(pairs_c)}")

        # ontology map: meds sharing a therapeutic class; icd/symptom codes
        # sharing an ICD chapter (first character of the code).
        concept_ontology = [[] for _ in range(n_concept_gids)]
        if self.graph_structure in ('ontology', 'full') and n_concept_gids > 1:
            try:
                from shared.data_prep.med_classes import MED_CLASS
            except Exception:
                MED_CLASS = {}

            def _chapter(code):
                code = str(code).strip()
                return code[0].upper() if code else None

            groups = {}   # ontology-class label -> list of concept gids
            for j, col in enumerate(med_cols):
                gen = col.split('_', 1)[1] if '_' in col else col
                cls = MED_CLASS.get(gen) or MED_CLASS.get(gen.lower())
                if cls:
                    groups.setdefault(('med', cls), []).append(CG_MED + j)
            for j, code in enumerate(icd_vocab):
                ch = _chapter(code)
                if ch:
                    groups.setdefault(('icd', ch), []).append(CG_ICD + j)
            for j, code in enumerate(symptom_vocab):
                ch = _chapter(code)
                if ch:
                    groups.setdefault(('sym', ch), []).append(CG_SYM + j)
            n_ont = 0
            for members in groups.values():
                for a_i in range(len(members)):
                    for b_i in range(a_i + 1, len(members)):
                        a, b = members[a_i], members[b_i]
                        concept_ontology[a].append(b)
                        concept_ontology[b].append(a)
                        n_ont += 1
            print(f"  Concept-concept ontology edges (same class/chapter): {n_ont}")

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

            # Concept nodes (med/icd/symptom/cc) also register their GLOBAL
            # concept id so the structure-driven edge builder below can wire
            # concept↔concept edges from the population cooccur/ontology maps.
            concept_local_of_gid = {}     # global concept id -> local node index

            # Med nodes
            med_offset = 1 + V_v
            for k, g_idx in enumerate(patient_meds):
                g_idx = int(g_idx)
                n = med_offset + k
                x[n, IDX_TYPE_MED] = 1.0
                x[n, IDX_MED_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0
                concept_local_of_gid[CG_MED + g_idx] = n

            # ICD diagnosis nodes
            icd_offset = 1 + V_v + num_med_nodes
            for k, g_idx in enumerate(patient_icds):
                n = icd_offset + k
                x[n, IDX_TYPE_ICD] = 1.0
                x[n, IDX_ICD_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0
                concept_local_of_gid[CG_ICD + g_idx] = n

            # SYMPTOM (doctor-refined ICD R-code) nodes
            sym_offset = 1 + V_v + num_med_nodes + num_icd_nodes
            for k, g_idx in enumerate(patient_syms):
                n = sym_offset + k
                x[n, IDX_TYPE_SYMPTOM] = 1.0
                x[n, IDX_SYM_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0
                concept_local_of_gid[CG_SYM + g_idx] = n

            # CHIEF-COMPLAINT (triage free-text) nodes
            cc_offset = 1 + V_v + num_med_nodes + num_icd_nodes + num_sym_nodes
            for k, g_idx in enumerate(patient_cc):
                n = cc_offset + k
                x[n, IDX_TYPE_CC] = 1.0
                x[n, IDX_CC_START + g_idx] = 1.0
                x[n, IDX_VALUE] = 1.0
                concept_local_of_gid[CG_CC + g_idx] = n

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
            # concept ↔ concept edges, governed by self.graph_structure.
            # (patient↔concept spokes above are always present; this only adds
            # the extra topology that distinguishes the structures.)
            struct = self.graph_structure
            if struct in ('cooccur', 'ontology', 'full'):
                nbr_maps = []
                if struct in ('cooccur', 'full'):
                    nbr_maps.append(concept_cooccur)
                if struct in ('ontology', 'full'):
                    nbr_maps.append(concept_ontology)
                seen_pairs = set()
                for gid, a in concept_local_of_gid.items():
                    for nbr_map in nbr_maps:
                        for gid_b in nbr_map[gid]:
                            if gid_b <= gid or gid_b not in concept_local_of_gid:
                                continue
                            if (gid, gid_b) in seen_pairs:
                                continue
                            seen_pairs.add((gid, gid_b))
                            b = concept_local_of_gid[gid_b]
                            src += [a, b]; dst += [b, a]
            # struct == 'star' -> no concept↔concept edges (pure hub baseline)
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
                "graph_structure": self.graph_structure,
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


def get_dataset(dataset_dir, dataset_name, task=None, graph_structure=None):
    if dataset_name.lower().startswith('mimic_intra_patient'):
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
        # Graph structure: explicit arg wins; otherwise fall back to a
        # trailing name token (e.g. mimic_intra_patient_cooccur) or 'star'.
        from shared.lib.graph_structures import all_structures
        struct = graph_structure
        if struct is None:
            struct = next((p for p in parts if p in all_structures()), 'star')
        return IntraPatientHeteroDataset(
            root=dataset_dir, name=dataset_name, csv_filename=csv,
            med_min_prev=prev, pmi_threshold=pmi, drop_los=not with_los,
            icd_min_prev=icd_prev, target=target, graph_structure=struct,
        )
    else:
        raise NotImplementedError


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

    # Aligned-comparison hook: if CANONICAL_SPLIT_JSON is set, take the exact
    # train/val/test folds from it (by subject_id) so ProtGNN and GraphCare are
    # evaluated on the IDENTICAL test patients. Graphs whose subject is not in
    # the canonical set (e.g. 0-code patients) are excluded from all folds.
    _canon = os.environ.get("CANONICAL_SPLIT_JSON")
    if _canon:
        fold = json.load(open(_canon))["fold"]
        groups = _extract_graph_groups(dataset, "subject_id")
        tr, ev, te = [], [], []
        for i in range(len(dataset)):
            f = fold.get(str(int(groups[i])))
            if f == 0:
                tr.append(i)
            elif f == 1:
                ev.append(i)
            elif f == 2:
                te.append(i)
        excluded = len(dataset) - len(tr) - len(ev) - len(te)
        print(f"  [split] CANONICAL: train={len(tr)} val={len(ev)} test={len(te)} (excluded {excluded})")
        return {
            'train': DataLoader(Subset(dataset, tr), batch_size=batch_size, shuffle=True),
            'eval':  DataLoader(Subset(dataset, ev), batch_size=batch_size, shuffle=False),
            'test':  DataLoader(Subset(dataset, te), batch_size=batch_size, shuffle=False),
        }

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
