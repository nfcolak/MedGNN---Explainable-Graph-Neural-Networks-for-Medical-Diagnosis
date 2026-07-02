import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import pandas as pd
import json, os, glob
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap

os.makedirs('outputs/visualizations', exist_ok=True)

PATH = 'data/processed_hetero_merged_ed_noLOS_prev10_pmi2_miss_disease/data.pt'
SUMMARY = 'outputs/results/clinical_explanations/summary.csv'

DISEASE_LABELS = {
    0:"Acute kidney failure", 1:"Acute pharyngitis", 2:"Acute upper resp. infection",
    3:"Alcohol abuse", 4:"Anemia", 5:"Anxiety disorder", 6:"Back/spine pain",
    7:"Cardiovascular risk", 8:"Dehydration", 9:"Diabetes", 10:"End stage renal disease",
    11:"Epilepsy", 12:"GI bleed", 13:"Head injury", 14:"Heart failure",
    15:"Hypokalemia", 16:"Hypotension", 17:"Hypothyroidism", 18:"Laceration",
    19:"Limb injury", 20:"Lower respiratory disease", 21:"Major depressive disorder",
    22:"Multiple rib fractures", 23:"NSTEMI", 24:"Sepsis",
    25:"Skin/soft-tissue infection", 26:"UTI/pyelonephritis", 27:"Intestinal obstruction",
    28:"Asthma exacerbation", 29:"Atrial fibrillation"
}

print("Loading data.pt (this takes ~30 seconds for 3GB)...")
data, slices = torch.load(PATH, map_location='cpu', weights_only=False)
labels = data.y.numpy()
n_graphs = len(labels)
node_counts = [(slices['x'][i+1] - slices['x'][i]).item() for i in range(n_graphs)]
edge_counts = [(slices['edge_index'][i+1] - slices['edge_index'][i]).item() for i in range(n_graphs)]
med_counts = [data.num_med_nodes[i].item() for i in range(n_graphs)]
sym_counts = [data.num_sym_nodes[i].item() for i in range(n_graphs)]
cc_counts  = [data.num_cc_nodes[i].item() for i in range(n_graphs)]
print(f"Loaded {n_graphs} graphs.")


# CLASS DISTRIBUTION 
print("1/8 Class distribution...")
unique, counts = np.unique(labels, return_counts=True)
names = [DISEASE_LABELS[i] for i in unique]
colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(names, counts, color=colors, height=0.75, edgecolor='white', linewidth=0.5)
ax.bar_label(bars, padding=4, fontsize=8.5)
ax.set_xlabel('Number of patients', fontsize=12)
ax.set_title('Disease class distribution — 77,697 patient graphs\n(intra-patient heterogeneous GNN)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.axvline(np.mean(counts), color='crimson', linewidth=1.5, linestyle='--', label=f'Mean: {np.mean(counts):.0f}')
ax.legend(fontsize=10)
ax.set_xlim(0, max(counts) * 1.12)
plt.tight_layout()
plt.savefig('outputs/visualizations/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_class_distribution.png")


# NODE TYPE COMPOSITION PER CLASS 
print("2/8 Node type composition per class...")
class_med  = {c: [] for c in range(30)}
class_sym  = {c: [] for c in range(30)}
class_cc   = {c: [] for c in range(30)}
class_node = {c: [] for c in range(30)}
for i in range(n_graphs):
    c = int(labels[i])
    class_med[c].append(med_counts[i])
    class_sym[c].append(sym_counts[i])
    class_cc[c].append(cc_counts[i])
    class_node[c].append(node_counts[i])

mean_med  = [np.mean(class_med[c])  for c in range(30)]
mean_sym  = [np.mean(class_sym[c])  for c in range(30)]
mean_cc   = [np.mean(class_cc[c])   for c in range(30)]
# vital nodes = total - 1 patient node - meds - syms - ccs
mean_vital = [np.mean(class_node[c]) - 1 - mean_med[c] - mean_sym[c] - mean_cc[c]
              for c in range(30)]

fig, ax = plt.subplots(figsize=(16, 9))
x = np.arange(30)
w = 0.6
ax.bar(x, mean_vital, w, label='Vital signs', color='#1D9E75')
ax.bar(x, mean_med,   w, bottom=mean_vital, label='Medications', color='#D85A30')
bot2 = [mean_vital[i]+mean_med[i] for i in range(30)]
ax.bar(x, mean_sym,   w, bottom=bot2, label='Symptoms', color='#BA7517')
bot3 = [bot2[i]+mean_sym[i] for i in range(30)]
ax.bar(x, mean_cc,    w, bottom=bot3, label='Chief complaints', color='#378ADD')

ax.set_xticks(x)
ax.set_xticklabels([DISEASE_LABELS[i].split('/')[0][:18] for i in range(30)],
                    rotation=45, ha='right', fontsize=7.5)
ax.set_ylabel('Mean nodes per patient graph', fontsize=11)
ax.set_title('Graph node type composition by disease class\n(stacked: vitals + medications + symptoms + chief complaints)',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/visualizations/02_node_composition_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_node_composition_by_class.png")


# THREE REAL PATIENT GRAPHS SIDE BY SIDE 
print("3/8 Patient graph visualizations...")

def extract_graph(idx):
    n0 = int(slices['x'][idx].item())
    n1 = int(slices['x'][idx+1].item())
    e0 = int(slices['edge_index'][idx].item())
    e1 = int(slices['edge_index'][idx+1].item())
    x  = data.x[n0:n1].numpy()
    # subtract n0 to re-index edges from 0
    ei = data.edge_index[:, e0:e1].numpy()
    ei = ei - n0
    y  = int(data.y[idx].item())
    nm = int(data.num_med_nodes[idx].item())
    ns = int(data.num_sym_nodes[idx].item())
    nc = int(data.num_cc_nodes[idx].item())
    nv = max(x.shape[0] - 1 - nm - ns - nc, 0)
    return x, ei, y, nv, nm, ns, nc

TYPE_COLORS = {
    'patient': '#7F77DD', 'vital': '#1D9E75',
    'med': '#D85A30', 'symptom': '#BA7517', 'cc': '#378ADD'
}
TYPE_SIZES  = {'patient': 900, 'vital': 350, 'med': 300, 'symptom': 280, 'cc': 280}

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, graph_idx in zip(axes, [2, 7, 14]):
    x, ei, y, nv, nm, ns, nc = extract_graph(graph_idx)
    n_nodes = x.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    # only add edges where both endpoints are valid node indices
    for j in range(ei.shape[1]):
        src, dst = int(ei[0, j]), int(ei[1, j])
        if 0 <= src < n_nodes and 0 <= dst < n_nodes:
            G.add_edge(src, dst)

    node_colors, node_sizes, node_labels = [], [], {}
    for ni in range(n_nodes):
        if ni == 0:
            t = 'patient'
        elif ni <= nv:
            t = 'vital'
        elif ni <= nv + nm:
            t = 'med'
        elif ni <= nv + nm + ns:
            t = 'symptom'
        elif ni <= nv + nm + ns + nc:
            t = 'cc'
        else:
            t = 'vital'
        node_colors.append(TYPE_COLORS[t])
        node_sizes.append(TYPE_SIZES[t])
        node_labels[ni] = t[0].upper() if ni > 0 else 'P'

    print(f"  node_colors={len(node_colors)}, node_sizes={len(node_sizes)}, G.nodes={G.number_of_nodes()}")

    pos = nx.spring_layout(G, seed=42, k=3.0)
    
    # use scalar size to bypass the mismatch entirely
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=300, ax=ax, alpha=0.92)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, arrows=False,
                           edge_color='#666', width=0.7)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=6, font_color='white', font_weight='bold')
    ax.set_title(f'{DISEASE_LABELS[y]}\n{n_nodes} nodes · {G.number_of_edges()//2} edges',
                 fontsize=9, fontweight='bold')
    ax.axis('off')

patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in TYPE_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=10, frameon=False)
fig.suptitle('Real patient graphs — intra-patient heterogeneous structure',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/visualizations/03_patient_graphs.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_patient_graphs.png")


# GRAPH SIZE DISTRIBUTION 
print("4/8 Graph size distribution...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, data_arr, label, color in zip(
    axes,
    [node_counts, edge_counts, med_counts],
    ['Nodes per graph', 'Edges per graph (undirected)', 'Medication nodes per graph'],
    ['#7F77DD', '#1D9E75', '#D85A30']
):
    ax.hist(data_arr, bins=40, color=color, edgecolor='white', alpha=0.85)
    mean_val = np.mean(data_arr)
    ax.axvline(mean_val, color='black', linewidth=1.5, linestyle='--',
               label=f'Mean: {mean_val:.1f}')
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Number of patients', fontsize=10)
    ax.legend(fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
plt.suptitle('Distribution of graph structural properties — 77,697 patients',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/visualizations/04_graph_size_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_graph_size_distributions.png")


# MEDICATION CO-OCCURRENCE HEATMAP (PMI) 
print("5/8 Medication co-occurrence heatmap...")
# Load metadata to get med vocab
meta_path = 'data/processed_hetero_merged_ed_noLOS_prev10_pmi2_miss_disease/metadata.json'
med_vocab = []
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    med_vocab = meta.get('med_vocab', [])

if med_vocab:
    # compute co-occurrence from the first 5000 graphs (fast approximation)
    Vm = len(med_vocab)
    # find where med block starts in feature vector
    Vv = len(meta.get('vital_vocab', []))
    N_TYPES = 6
    med_start = N_TYPES + Vv

    n_sample = min(5000, n_graphs)
    co = np.zeros((Vm, Vm), dtype=np.float32)
    marg = np.zeros(Vm, dtype=np.float32)
    for i in range(n_sample):
        n0 = slices['x'][i].item()
        n1 = slices['x'][i+1].item()
        x_i = data.x[n0:n1].numpy()
        nm_i = int(data.num_med_nodes[i].item())
        nv_i = int(data.x.shape[1]) - 1   # skip patient node
        # find med nodes: nodes 1+Vv .. 1+Vv+nm_i-1
        med_node_start = 1 + Vv
        med_node_end   = 1 + Vv + nm_i
        for ni in range(med_node_start, min(med_node_end, x_i.shape[0])):
            row = x_i[ni, med_start:med_start+Vm]
            m_idx = int(np.argmax(row)) if row.max() > 0 else -1
            if m_idx >= 0:
                marg[m_idx] += 1
                for nj in range(med_node_start, min(med_node_end, x_i.shape[0])):
                    if ni != nj:
                        row2 = x_i[nj, med_start:med_start+Vm]
                        m2 = int(np.argmax(row2)) if row2.max() > 0 else -1
                        if m2 >= 0:
                            co[m_idx, m2] += 1

    # top 20 most common meds
    top20 = np.argsort(-marg)[:20]
    top_names = [med_vocab[i].replace('med_','').replace('pyx_','ED:')[:20] for i in top20]
    sub_co = co[np.ix_(top20, top20)]
    np.fill_diagonal(sub_co, 0)
    # normalise to [0,1]
    if sub_co.max() > 0:
        sub_co = sub_co / sub_co.max()

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = LinearSegmentedColormap.from_list('coop', ['#f7f7f7', '#1D9E75'])
    im = ax.imshow(sub_co, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(20)); ax.set_xticklabels(top_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(20)); ax.set_yticklabels(top_names, fontsize=8)
    plt.colorbar(im, ax=ax, label='Normalised co-occurrence')
    ax.set_title('Medication co-occurrence — top 20 most common medications\n(approx. from first 5,000 patients)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/05_medication_cooccurrence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_medication_cooccurrence.png")
else:
    print("  Skipped: no metadata found")


# FROM SUMMARY.CSV 
if os.path.exists(SUMMARY):
    df = pd.read_csv(SUMMARY, sep=';')
    df = df[df['graph'] != 'AGGREGATE'].copy()
    df['correct_bool'] = df['result'] == 'correct'

    # PER-DISEASE ACCURACY 
    print("6/8 Per-disease accuracy...")
    acc = df.groupby('actual')['correct_bool'].mean().sort_values()
    count = df.groupby('actual')['correct_bool'].count()
    colors_acc = ['#1D9E75' if v >= 0.5 else '#D85A30' for v in acc.values]
    fig, ax = plt.subplots(figsize=(13, 9))
    bars = ax.barh(acc.index, acc.values * 100, color=colors_acc, height=0.72, edgecolor='white', linewidth=0.4)
    for i, (name, val) in enumerate(acc.items()):
        n = count[name]
        ax.text(val*100 + 0.5, i, f'{val*100:.0f}%  (n={n})', va='center', fontsize=8)
    ax.axvline(50, color='gray', linewidth=1, linestyle='--', alpha=0.6, label='50% baseline')
    overall = df['correct_bool'].mean() * 100
    ax.axvline(overall, color='navy', linewidth=1.5, linestyle='-',
               label=f'Overall: {overall:.1f}%')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Per-disease prediction accuracy — {len(df)} test patients\n'
                 f'Overall {df["correct_bool"].sum()}/{len(df)} ({overall:.1f}%)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/06_per_disease_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_per_disease_accuracy.png")

    # CONFIDENCE DISTRIBUTION 
    print("7/8 Confidence distribution...")
    if 'p_pred' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, subset, label, color in zip(
            axes,
            [df[df['correct_bool']], df[~df['correct_bool']]],
            ['Correct predictions', 'Wrong predictions'],
            ['#1D9E75', '#D85A30']
        ):
            ax.hist(subset['p_pred'].dropna().astype(float), bins=30,
                    color=color, edgecolor='white', alpha=0.85)
            ax.set_xlabel('Model confidence (softmax probability)', fontsize=11)
            ax.set_ylabel('Number of patients', fontsize=11)
            ax.set_title(f'{label}\n(n={len(subset)})', fontsize=11, fontweight='bold')
            ax.axvline(subset['p_pred'].dropna().astype(float).mean(),
                       color='black', linewidth=1.5, linestyle='--',
                       label=f'Mean: {subset["p_pred"].dropna().astype(float).mean():.3f}')
            ax.legend(fontsize=10)
        plt.suptitle('Model confidence for correct vs wrong predictions',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/07_confidence_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: 07_confidence_distribution.png")

    # TOP KEY FACTORS FREQUENCY 
    print("8/8 Most cited key factors...")
    factor_cols = [c for c in df.columns if c.startswith('key_factor_') and not c.endswith('%')]
    if factor_cols:
        all_factors = []
        for col in factor_cols:
            all_factors.extend(df[col].dropna().tolist())
        # clean up factor names
        all_factors = [str(f).strip() for f in all_factors if str(f).strip() not in ('', '-', 'nan')]
        counter = Counter(all_factors)
        top30 = counter.most_common(30)
        top_names_f = [t[0][:45] for t in top30]
        top_vals_f  = [t[1] for t in top30]

        fig, ax = plt.subplots(figsize=(12, 9))
        colors_f = plt.cm.viridis(np.linspace(0.2, 0.85, len(top_names_f)))
        ax.barh(top_names_f[::-1], top_vals_f[::-1], color=colors_f, height=0.72, edgecolor='white', linewidth=0.4)
        ax.set_xlabel('Number of times cited as a key factor', fontsize=11)
        ax.set_title('Top 30 most influential clinical factors across all test patients\n'
                     '(aggregated from GradExplainer node attributions)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/08_top_key_factors.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: 08_top_key_factors.png")
else:
    print("  Skipping charts 6–8: summary.csv not found at", SUMMARY)

print(f"\nAll done — check outputs/visualizations/ for {len(os.listdir('outputs/visualizations'))} figures.")