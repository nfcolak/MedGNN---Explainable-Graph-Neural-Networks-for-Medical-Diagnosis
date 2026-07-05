import os
import re
import csv
import glob
import json
from collections import Counter, defaultdict
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
 
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
OUT = os.path.join(_ROOT, "comparison", "visualizations")
os.makedirs(OUT, exist_ok=True)
 
# ---- unified theme (teal / orange) ----
TEAL, ORANGE, INK, MUTE = "#1D9E75", "#D85A30", "#12233A", "#8A97A6"
TYPE_COLORS = {"patient": "#7F77DD", "vital": TEAL, "med": ORANGE, "symptom": "#BA7517", "cc": "#378ADD"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#CCD3DB"})
 
DATA_PT = os.path.join(_ROOT, "data", "processed_hetero_merged_ed_noLOS_prev10_pmi2_miss_disease", "data.pt")
META = os.path.join(_ROOT, "data", "processed_hetero_merged_ed_noLOS_prev10_pmi2_miss_disease", "metadata.json")
GC = os.path.join(_ROOT, "graphcare_analysis")
PG_RUN = os.path.join(_ROOT, "protgnn_analysis", "outputs", "results", "2026-07-03_11-01_disease_gcn")
PG_CLINICAL = os.path.join(_ROOT, "protgnn_analysis", "outputs", "results", "clinical_explanations", "summary.csv")
 
DISEASE_LABELS = {
    0:"Acute kidney failure",1:"Acute pharyngitis",2:"Acute upper resp. infection",3:"Alcohol abuse",
    4:"Anemia",5:"Anxiety disorder",6:"Back/spine pain",7:"Cardiovascular risk",8:"Dehydration",
    9:"Diabetes",10:"End stage renal disease",11:"Epilepsy",12:"GI bleed",13:"Head injury",
    14:"Heart failure",15:"Hypokalemia",16:"Hypotension",17:"Hypothyroidism",18:"Laceration",
    19:"Limb injury",20:"Lower respiratory disease",21:"Major depressive disorder",
    22:"Multiple rib fractures",23:"NSTEMI",24:"Sepsis",25:"Skin/soft-tissue infection",
    26:"UTI/pyelonephritis",27:"Intestinal obstruction",28:"Asthma exacerbation",29:"Atrial fibrillation"
}
EXPLAINERS = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]
SHORT = {"GradExplainer":"Grad","IntegratedGradExplainer":"IntGrad","GNNExplainer":"GNNExp"}
 
 
# ══════════════════════════ PART A — dataset (from data.pt) ══════════════════════════
def part_a():
    if not os.path.isfile(DATA_PT):
        print("[SKIP] Part A — data.pt not found at", DATA_PT); return
    import torch
    print("Loading data.pt (~30s for 3GB)...")
    data, slices = torch.load(DATA_PT, map_location="cpu", weights_only=False)
    labels = data.y.numpy(); n = len(labels)
    node_counts = [(slices["x"][i+1]-slices["x"][i]).item() for i in range(n)]
    edge_counts = [(slices["edge_index"][i+1]-slices["edge_index"][i]).item() for i in range(n)]
    med = [data.num_med_nodes[i].item() for i in range(n)]
    sym = [data.num_sym_nodes[i].item() for i in range(n)]
    cc  = [data.num_cc_nodes[i].item() for i in range(n)]
    print(f"  loaded {n} graphs")
 
    # 01 class distribution
    uniq, cnt = np.unique(labels, return_counts=True)
    names = [DISEASE_LABELS[i] for i in uniq]
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(names, cnt, color=plt.cm.tab20(np.linspace(0,1,len(uniq))), height=0.75, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=4, fontsize=8.5)
    ax.set_xlabel("Number of patients"); ax.invert_yaxis()
    ax.axvline(np.mean(cnt), color="crimson", ls="--", lw=1.5, label=f"Mean: {np.mean(cnt):.0f}")
    ax.set_title("Disease class distribution — 77,697 patient graphs", fontweight="bold")
    ax.legend(); ax.set_xlim(0, max(cnt)*1.12)
    plt.tight_layout(); plt.savefig(f"{OUT}/01_class_distribution.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  saved 01_class_distribution.png")
 
    # 02 node-type composition per class
    cm={c:[] for c in range(30)}; cs={c:[] for c in range(30)}; cc2={c:[] for c in range(30)}; cn={c:[] for c in range(30)}
    for i in range(n):
        c=int(labels[i]); cm[c].append(med[i]); cs[c].append(sym[i]); cc2[c].append(cc[i]); cn[c].append(node_counts[i])
    mm=[np.mean(cm[c]) for c in range(30)]; ms=[np.mean(cs[c]) for c in range(30)]
    mc=[np.mean(cc2[c]) for c in range(30)]; mv=[np.mean(cn[c])-1-mm[c]-ms[c]-mc[c] for c in range(30)]
    fig, ax = plt.subplots(figsize=(16,9)); x=np.arange(30); w=0.6
    ax.bar(x,mv,w,label="Vital signs",color=TYPE_COLORS["vital"])
    ax.bar(x,mm,w,bottom=mv,label="Medications",color=TYPE_COLORS["med"])
    b2=[mv[i]+mm[i] for i in range(30)]; ax.bar(x,ms,w,bottom=b2,label="Symptoms",color=TYPE_COLORS["symptom"])
    b3=[b2[i]+ms[i] for i in range(30)]; ax.bar(x,mc,w,bottom=b3,label="Chief complaints",color=TYPE_COLORS["cc"])
    ax.set_xticks(x); ax.set_xticklabels([DISEASE_LABELS[i].split('/')[0][:18] for i in range(30)],rotation=45,ha="right",fontsize=7.5)
    ax.set_ylabel("Mean nodes per patient graph"); ax.legend(loc="upper right")
    ax.set_title("Graph node-type composition by disease class", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/02_node_composition_by_class.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  saved 02_node_composition_by_class.png")
 
    # 03 patient graphs  (kept as-is per user; guarded so a bug won't kill the run)
    try:
        import networkx as nx
        def extract(idx):
            n0=int(slices["x"][idx].item()); n1=int(slices["x"][idx+1].item())
            e0=int(slices["edge_index"][idx].item()); e1=int(slices["edge_index"][idx+1].item())
            x=data.x[n0:n1].numpy(); ei=data.edge_index[:,e0:e1].numpy()-n0
            y=int(data.y[idx].item()); nm=int(data.num_med_nodes[idx].item())
            ns=int(data.num_sym_nodes[idx].item()); nc=int(data.num_cc_nodes[idx].item())
            nv=max(x.shape[0]-1-nm-ns-nc,0); return x,ei,y,nv,nm,ns,nc
        fig, axes = plt.subplots(1,3,figsize=(18,7))
        for ax, gi in zip(axes, [2,7,14]):
            x,ei,y,nv,nm,ns,nc = extract(gi); nn=x.shape[0]
            G=nx.DiGraph(); G.add_nodes_from(range(nn))
            for j in range(ei.shape[1]):
                s,d=int(ei[0,j]),int(ei[1,j])
                if 0<=s<nn and 0<=d<nn: G.add_edge(s,d)
            colors=[]; labs={}
            for ni in range(nn):
                t=("patient" if ni==0 else "vital" if ni<=nv else "med" if ni<=nv+nm
                   else "symptom" if ni<=nv+nm+ns else "cc" if ni<=nv+nm+ns+nc else "vital")
                colors.append(TYPE_COLORS[t]); labs[ni]=("P" if ni==0 else t[0].upper())
            pos=nx.spring_layout(G, seed=42, k=3.0)
            nx.draw_networkx_nodes(G,pos,node_color=colors,node_size=300,ax=ax,alpha=0.92)
            nx.draw_networkx_edges(G,pos,ax=ax,alpha=0.25,arrows=False,edge_color="#666",width=0.7)
            nx.draw_networkx_labels(G,pos,labels=labs,ax=ax,font_size=6,font_color="white",font_weight="bold")
            ax.set_title(f"{DISEASE_LABELS[y]}\n{nn} nodes · {G.number_of_edges()//2} edges",fontsize=9,fontweight="bold")
            ax.axis("off")
        fig.legend(handles=[mpatches.Patch(color=v,label=k.capitalize()) for k,v in TYPE_COLORS.items()],
                   loc="lower center", ncol=5, fontsize=10, frameon=False)
        fig.suptitle("Real patient graphs — intra-patient heterogeneous structure",fontsize=13,fontweight="bold",y=1.01)
        plt.tight_layout(); plt.savefig(f"{OUT}/03_patient_graphs.png", dpi=150, bbox_inches="tight"); plt.close()
        print("  saved 03_patient_graphs.png")
    except Exception as e:
        print(f"  [SKIP] 03_patient_graphs — {type(e).__name__}: {e}")
 
    # 04 graph-size distributions
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    for ax, arr, lab, col in zip(axes,[node_counts,edge_counts,med],
                                 ["Nodes per graph","Edges per graph (undirected)","Medication nodes per graph"],
                                 [TYPE_COLORS["patient"],TEAL,ORANGE]):
        ax.hist(arr,bins=40,color=col,edgecolor="white",alpha=0.85)
        ax.axvline(np.mean(arr),color="black",ls="--",lw=1.5,label=f"Mean: {np.mean(arr):.1f}")
        ax.set_xlabel(lab); ax.set_ylabel("Number of patients"); ax.legend(); ax.set_title(lab,fontweight="bold")
    plt.suptitle("Graph structural properties — 77,697 patients",fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_graph_size_distributions.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  saved 04_graph_size_distributions.png")
 
    # 05 medication co-occurrence
    try:
        med_vocab=[]
        if os.path.isfile(META):
            meta=json.load(open(META)); med_vocab=meta.get("med_vocab",[])
        if med_vocab:
            Vm=len(med_vocab); Vv=len(meta.get("vital_vocab",[])); med_start=6+Vv
            ns_=min(5000,n); co=np.zeros((Vm,Vm),np.float32); marg=np.zeros(Vm,np.float32)
            for i in range(ns_):
                n0=slices["x"][i].item(); n1=slices["x"][i+1].item(); xi=data.x[n0:n1].numpy()
                nm_i=int(data.num_med_nodes[i].item()); s0=1+Vv; s1=1+Vv+nm_i
                idxs=[]
                for ni in range(s0,min(s1,xi.shape[0])):
                    row=xi[ni,med_start:med_start+Vm]; mi=int(np.argmax(row)) if row.max()>0 else -1
                    if mi>=0: idxs.append(mi); marg[mi]+=1
                for a in idxs:
                    for b in idxs:
                        if a!=b: co[a,b]+=1
            top=np.argsort(-marg)[:20]
            names=[med_vocab[i].replace("med_","").replace("pyx_","ED:")[:20] for i in top]
            sub=co[np.ix_(top,top)]; np.fill_diagonal(sub,0)
            if sub.max()>0: sub=sub/sub.max()
            fig,ax=plt.subplots(figsize=(12,10))
            im=ax.imshow(sub,cmap=LinearSegmentedColormap.from_list("c",["#f7f7f7",TEAL]),vmin=0,vmax=1)
            ax.set_xticks(range(20)); ax.set_xticklabels(names,rotation=45,ha="right",fontsize=8)
            ax.set_yticks(range(20)); ax.set_yticklabels(names,fontsize=8)
            plt.colorbar(im,ax=ax,label="Normalised co-occurrence")
            ax.set_title("Medication co-occurrence — top 20 (first 5,000 patients)",fontweight="bold")
            plt.tight_layout(); plt.savefig(f"{OUT}/05_medication_cooccurrence.png", dpi=150, bbox_inches="tight"); plt.close()
            print("  saved 05_medication_cooccurrence.png")
        else:
            print("  [SKIP] 05 co-occurrence — no med_vocab in metadata")
    except Exception as e:
        print(f"  [SKIP] 05_medication_cooccurrence — {type(e).__name__}: {e}")
 
 
# ══════════════════════════ PART B — results comparison ══════════════════════════
def _report_metrics(path):
    if not os.path.isfile(path): return {}
    m={}
    for line in open(path):
        mm=re.match(r"\s*(accuracy|acc|balanced_acc|macro_f1|micro_f1|top3_acc|top5_acc)\s*[:=]\s*([\d.]+)",line)
        if mm:
            k="accuracy" if mm.group(1)=="acc" else mm.group(1); m[k]=float(mm.group(2))
    return m
 
def _sparsity(imp, mass=0.9):
    a=np.abs(np.asarray(imp,float)); nn=len(a)
    if nn==0 or a.sum()==0: return 0.0
    c=np.cumsum(np.sort(a)[::-1])/a.sum(); return 1.0-(int(np.searchsorted(c,mass)+1))/nn
 
def _spar_jsons(d):
    files=glob.glob(os.path.join(d,"graph_*.json"))
    if not files: return None
    agg={e:[] for e in EXPLAINERS}; nag=nat=0
    for f in files:
        ex=json.load(open(f)).get("explanations",{}); ti={}
        for e in EXPLAINERS:
            imp=ex.get(e,{}).get("node_importance")
            if imp is None: continue
            agg[e].append(_sparsity(imp)); ti[e]=int(np.argmax(np.abs(imp)))
        v=list(ti.values())
        if len(v)>=2: nat+=1; nag+=int(len(set(v))==1)
    return {"sp":{e:(float(np.mean(agg[e])) if agg[e] else None) for e in EXPLAINERS},
            "ag":(nag/nat if nat else None),"n":len(files),"raw":(nag,nat)}
 
def _norm(t):
    t=t.strip(); m=re.match(r"^[a-zA-Z_]+\[(.+)\]$",t)
    if m: t=m.group(1)
    t=re.sub(r"^(ED:|med_|pyx_)","",t); t=re.sub(r"[↑↓]","",t)
    return t.strip().lower()
 
def part_b():
    # 06 model performance
    pg=_report_metrics(os.path.join(PG_RUN,"report.txt")); gc=_report_metrics(os.path.join(GC,"outputs","report.txt"))
    if pg and gc:
        mets=[(k,l) for k,l in [("accuracy","Accuracy"),("balanced_acc","Balanced acc"),
              ("macro_f1","Macro-F1"),("top3_acc","Top-3"),("top5_acc","Top-5")] if k in pg and k in gc]
        x=np.arange(len(mets)); w=0.38
        fig,ax=plt.subplots(figsize=(10,5.2))
        b1=ax.bar(x-w/2,[pg[k] for k,_ in mets],w,label="ProtGNN (GCN + prototypes)",color=TEAL)
        b2=ax.bar(x+w/2,[gc[k] for k,_ in mets],w,label="GraphCare (BAT-GNN)",color=ORANGE)
        for b in (b1,b2): ax.bar_label(b,fmt="%.3f",padding=3,fontsize=9,color=INK)
        ax.set_xticks(x); ax.set_xticklabels([l for _,l in mets]); ax.set_ylim(0,1); ax.set_ylabel("Score")
        ax.set_title("Model performance — full test set (30-class ED diagnosis)",fontweight="bold",color=INK)
        ax.legend(frameon=False); ax.grid(axis="y",color="#EEF1F4"); ax.set_axisbelow(True)
        plt.tight_layout(); plt.savefig(f"{OUT}/06_model_performance.png",dpi=150,bbox_inches="tight"); plt.close()
        print("  saved 06_model_performance.png")
    else:
        print("[SKIP] 06 model performance — missing report.txt")
 
    # 07 explainer sparsity
    pgs=_spar_jsons(os.path.join(PG_RUN,"explanations")); gcs=_spar_jsons(os.path.join(GC,"outputs","results","explanations"))
    if pgs and gcs:
        x=np.arange(3); w=0.38
        fig,ax=plt.subplots(figsize=(9,5.4))
        b1=ax.bar(x-w/2,[pgs["sp"][e] or 0 for e in EXPLAINERS],w,label=f"ProtGNN (n={pgs['n']})",color=TEAL)
        b2=ax.bar(x+w/2,[gcs["sp"][e] or 0 for e in EXPLAINERS],w,label=f"GraphCare (n={gcs['n']})",color=ORANGE)
        for b in (b1,b2): ax.bar_label(b,fmt="%.3f",padding=3,fontsize=9,color=INK)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[e] for e in EXPLAINERS])
        ax.set_ylabel("Mean sparsity  (higher = more concise)")
        ax.set_title("Explanation sparsity by explainer",fontweight="bold",color=INK)
        ax.legend(frameon=False); ax.grid(axis="y",color="#EEF1F4"); ax.set_axisbelow(True)
        ap,ag=pgs["raw"],gcs["raw"]
        ax.text(0.5,-0.22,f"Top-node agreement — ProtGNN {ap[0]}/{ap[1]} ({(pgs['ag'] or 0):.0%})   |   "
                f"GraphCare {ag[0]}/{ag[1]} ({(gcs['ag'] or 0):.0%})",transform=ax.transAxes,ha="center",fontsize=9,color=MUTE)
        ax.text(0.5,-0.30,"Distributional (unmatched) — same canonical split, 50 graphs each, not patient-identical",
                transform=ax.transAxes,ha="center",fontsize=8,color=MUTE,style="italic")
        plt.tight_layout(); plt.savefig(f"{OUT}/07_explainer_sparsity.png",dpi=150,bbox_inches="tight"); plt.close()
        print("  saved 07_explainer_sparsity.png")
    else:
        print("[SKIP] 07 sparsity — missing explanation JSONs")
 
    # 08 clinical-factor overlap
    def gc_tokens():
        p=os.path.join(GC,"outputs","results","graphxai_summary.csv")
        if not os.path.isfile(p): return None
        out=[]
        for row in csv.DictReader(open(p),delimiter=";"):
            if row.get("graph")=="AGGREGATE": continue
            out+=[t.strip() for t in (row.get("grad_top_factors","") or "").split(",") if t.strip() and not t.startswith("(error")]
        return out
    def pg_tokens():
        if not os.path.isfile(PG_CLINICAL): return None
        import pandas as pd
        df=pd.read_csv(PG_CLINICAL,sep=";"); cols=[c for c in df.columns if c.startswith("key_factor_") and not c.endswith("%")]
        out=[]
        for c in cols: out+=[str(v).strip() for v in df[c].dropna().tolist() if str(v).strip() not in ("","-","nan")]
        return out
    gt,pt=gc_tokens(),pg_tokens()
    if gt and pt:
        gc_c=Counter(_norm(t) for t in gt); pg_c=Counter(_norm(t) for t in pt)
        shared=set(gc_c)&set(pg_c); top=sorted(shared,key=lambda k:gc_c[k]+pg_c[k],reverse=True)[:15]
        if top:
            y=np.arange(len(top))[::-1]
            fig,ax=plt.subplots(figsize=(10,6.5))
            ax.barh(y+0.19,[pg_c[t] for t in top],0.38,label="ProtGNN",color=TEAL)
            ax.barh(y-0.19,[gc_c[t] for t in top],0.38,label="GraphCare",color=ORANGE)
            ax.set_yticks(y); ax.set_yticklabels(top); ax.set_xlabel("Times cited as a top factor")
            ax.set_title("Clinical factors surfaced by BOTH methods (top 15)",fontweight="bold",color=INK,fontsize=12)
            ax.legend(frameon=False); ax.grid(axis="x",color="#EEF1F4"); ax.set_axisbelow(True)
            ax.text(0.98,0.02,f"{len(shared)} shared · {len(set(pg_c)-shared)} ProtGNN-only · {len(set(gc_c)-shared)} GraphCare-only",
                    transform=ax.transAxes,ha="right",fontsize=8,color=MUTE)
            plt.tight_layout(); plt.savefig(f"{OUT}/08_factor_overlap.png",dpi=150,bbox_inches="tight"); plt.close()
            print("  saved 08_factor_overlap.png")
        else:
            print("[SKIP] 08 factor overlap — no shared tokens")
    else:
        print("[SKIP] 08 factor overlap — missing decoded factors (pg:%s gc:%s)"%(bool(pt),bool(gt)))
 
    # 09 per-class factors (GraphCare)
    p=os.path.join(GC,"outputs","results","graphxai_summary.csv")
    if os.path.isfile(p):
        byc=defaultdict(Counter)
        for row in csv.DictReader(open(p),delimiter=";"):
            if row.get("graph")=="AGGREGATE": continue
            cls=row.get("actual","").strip()
            for t in (row.get("grad_top_factors","") or "").split(","):
                t=t.strip()
                if t and not t.startswith("(error"): byc[cls][_norm(t)]+=1
        classes=sorted([c for c in byc if c],key=lambda c:sum(byc[c].values()),reverse=True)[:12]
        if classes:
            fig,ax=plt.subplots(figsize=(12,max(5,0.55*len(classes)))); yl=[]
            for i,c in enumerate(classes[::-1]):
                yl.append(c[:22]); ax.text(0.01,i,", ".join(t for t,_ in byc[c].most_common(3)),va="center",fontsize=9,color=INK)
            ax.set_yticks(range(len(classes))); ax.set_yticklabels(yl,fontsize=9)
            ax.set_xticks([]); ax.set_xlim(0,1); ax.grid(False)
            for s in ("top","right","bottom"): ax.spines[s].set_visible(False)
            ax.set_title("Top explanation factors per disease class — GraphCare / GradExplainer\n(50-graph subset — partial coverage)",
                         fontweight="bold",color=INK,fontsize=12)
            plt.tight_layout(); plt.savefig(f"{OUT}/09_per_class_factors_graphcare.png",dpi=150,bbox_inches="tight"); plt.close()
            print("  saved 09_per_class_factors_graphcare.png  [50-graph subset]")
        else:
            print("[SKIP] 09 per-class factors — no classes parsed")
    else:
        print("[SKIP] 09 per-class factors — missing GraphCare summary")
 
    # 10 per-disease accuracy (guarded)
    fpg=os.path.join(PG_RUN,"per_disease_accuracy.csv"); fgc=os.path.join(GC,"outputs","per_disease_accuracy.csv")
    if os.path.isfile(fpg) and os.path.isfile(fgc):
        print("  [10] full-test per-disease CSVs found — (render path)")
    else:
        print("[INSUFFICIENT DATA] 10 per-disease accuracy — needs FULL-test per_disease_accuracy.csv "
              "for each model; the 50-graph subset (1-2 patients/class) would be noise. Skipped.")
 
 
if __name__ == "__main__":
    print("=== PART A: dataset characterization ===")
    part_a()
    print("=== PART B: results comparison ===")
    part_b()
    print("Done ->", OUT)
 