#graphxai_integration.py
"""
GraphXAI Integration for ProtGNN
=================================
Bu script, egitilmis ProtGNN modelini GraphXAI aciklayici kutuphanesiyle
entegre eder.  Her bir test grafigi icin asagidaki aciklayicilar calistirilir:

  - GradExplainer           (Vanilla Gradient)
  - IntegratedGradExplainer (Integrated Gradients)
  - GNNExplainer            (Learnable edge masks)

Calistirmak icin:
    python -m protgnn_analysis.explainability.graphxai_integration

Gereksinimler:
    GraphXAI paketinin Python yolunda olmasi gerekir.  Proje kokleri
    otomatik olarak sys.path'e eklenir; yoksa kurulum icin:
        pip install -e <GraphXAI-root>
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Yol ayarlari: GraphXAI kutuphanesini import edebilmek icin
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_GRAPHXAI_ROOT = os.path.join(_PROJECT_ROOT, "external", "GraphXAI-main")

for _p in [_PROJECT_ROOT, _SRC_DIR, _GRAPHXAI_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# ProtGNN imports
# ---------------------------------------------------------------------------
from protgnn_analysis.config import data_args, model_args, train_args
from protgnn_analysis.models import GnnNets
from protgnn_analysis.explainability.graphxai_wrapper import ProtGNNWrapper
from protgnn_analysis.load_dataset import get_dataset, get_dataloader

# ---------------------------------------------------------------------------
# GraphXAI imports
# ---------------------------------------------------------------------------
try:
    # Dogrudan modul yoluyla import ediyoruz; bu sayede PGMExplainer'in
    # Python-3.9-uyumsuz pgmpy bagimliligini tetiklemiyoruz.
    from graphxai.explainers.grad import GradExplainer
    from graphxai.explainers.integrated_grad import IntegratedGradExplainer
    from graphxai.explainers.gnn_explainer import GNNExplainer
    from graphxai.utils import Explanation
    GRAPHXAI_AVAILABLE = True
except ImportError as exc:
    print(f"[UYARI] GraphXAI yuklenemedi: {exc}")
    print("       GraphXAI'nin Python yolunda oldugunu dogrulayin.")
    GRAPHXAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

def load_trained_model(input_dim: int, output_dim: int) -> GnnNets:
    """Checkpoint'ten egitilmis GnnNets modelini yukler."""
    gnn_nets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = os.path.join(model_args.checkpoint, data_args.dataset_name)
    ckpt_path = os.path.join(ckpt_dir, f"{model_args.model_name}_best.pth")

    if not os.path.isfile(ckpt_path):
        print(f"[BILGI] Checkpoint bulunamadi ({ckpt_path}). Rastgele agirliklar kullanilacak.")
        gnn_nets.to_device()
        return gnn_nets

    checkpoint = torch.load(ckpt_path, map_location=model_args.device)
    gnn_nets.update_state_dict(checkpoint["net"])
    gnn_nets.to_device()
    gnn_nets.eval()
    print(f"[BILGI] Model yuklendi: {ckpt_path}  (epoch={checkpoint.get('epoch', '?')}, acc={checkpoint.get('acc', '?'):.4f})")
    return gnn_nets


def print_node_importance(exp: "Explanation", graph_idx: int, explainer_name: str) -> None:
    """Bir Explanation nesnesinin node onem skorlarini yazdirir."""
    if exp.node_imp is None:
        print(f"  [{explainer_name}] Graf {graph_idx}: node_imp yok")
        return

    imp = exp.node_imp.detach().cpu()
    print(
        f"  [{explainer_name}] Graf {graph_idx}: "
        f"node_imp shape={tuple(imp.shape)}, "
        f"min={imp.min():.4f}, max={imp.max():.4f}, "
        f"mean={imp.mean():.4f}"
    )


# ---------------------------------------------------------------------------
# Ana entegrasyon akisi
# ---------------------------------------------------------------------------

def run_graphxai_integration(num_test_graphs: int = 5) -> None:
    """
    Egitilmis ProtGNN modelini GraphXAI aciklayicilariyla calistirir.

    Args:
        num_test_graphs: Aciklama uretilecek test grafigi sayisi.
    """
    if not GRAPHXAI_AVAILABLE:
        print("GraphXAI yuklu degil; entegrasyon atlandi.")
        return

    # ------------------------------------------------------------------
    # 1. Veri seti ve model
    # ------------------------------------------------------------------
    print("=== Veri seti yukleniyor... ===")
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    print(f"    Dataset: {data_args.dataset_name}, input_dim={input_dim}, output_dim={output_dim}")

    dataloader = get_dataloader(
        dataset,
        batch_size=1,                       # Aciklama icin birer birer isliyoruz
        random_split_flag=data_args.random_split,
        data_split_ratio=data_args.data_split_ratio,
        seed=data_args.seed,
    )

    # ------------------------------------------------------------------
    # 2. Model yukle ve sar
    # ------------------------------------------------------------------
    print("\n=== Model yukleniyor... ===")
    gnn_nets = load_trained_model(input_dim, output_dim)

    # GraphXAI'nin beklentisine uygun sarmalayici
    wrapper = ProtGNNWrapper(gnn_nets)
    wrapper.eval()

    # ------------------------------------------------------------------
    # 3. GraphXAI aciklayicilarini baslat
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    grad_exp   = GradExplainer(wrapper, criterion=criterion)
    integ_exp  = IntegratedGradExplainer(wrapper, criterion=criterion)
    gnn_exp    = GNNExplainer(wrapper)

    print("\n=== GraphXAI Aciklayicilar baslatildi ===")
    print(f"    GradExplainer         : katman sayisi L={grad_exp.L}")
    print(f"    IntegratedGradExplainer: katman sayisi L={integ_exp.L}")

    # ------------------------------------------------------------------
    # 4. Test grafikleri uzerinde aciklama uret
    # ------------------------------------------------------------------
    print(f"\n=== {num_test_graphs} test grafigi aciklaniyor... ===\n")

    results = []
    processed = 0

    for batch in dataloader["test"]:
        if processed >= num_test_graphs:
            break

        # batch_size=1 oldugu icin tek grafik geliyor
        data = batch
        x          = data.x.to(model_args.device)
        edge_index = data.edge_index.to(model_args.device)
        label      = data.y.to(model_args.device)

        # Tek grafik icin batch vektoru: hepsi 0
        null_batch = torch.zeros(x.size(0), dtype=torch.long, device=model_args.device)
        forward_kwargs = {"batch": null_batch}

        # Model tahmini
        with torch.no_grad():
            logits = wrapper(x, edge_index, null_batch)
            pred_class = logits.argmax(dim=-1).item()

        print(f"Graf {processed} | gercek={label.item()}, tahmin={pred_class}, dugum_sayisi={x.size(0)}, kenar_sayisi={edge_index.size(1)}")

        graph_results = {"graph_idx": processed, "label": label.item(), "pred": pred_class}

        # --- GradExplainer ---
        try:
            exp_grad = grad_exp.get_explanation_graph(
                x=x,
                edge_index=edge_index,
                label=label,
                forward_kwargs=forward_kwargs,
            )
            print_node_importance(exp_grad, processed, "GradExplainer")
            graph_results["grad"] = exp_grad
        except Exception as e:
            print(f"  [GradExplainer] HATA: {e}")

        # --- IntegratedGradExplainer ---
        try:
            exp_ig = integ_exp.get_explanation_graph(
                x=x,
                edge_index=edge_index,
                label=label,
                forward_kwargs=forward_kwargs,
            )
            print_node_importance(exp_ig, processed, "IntegratedGrad")
            graph_results["integrated_grad"] = exp_ig
        except Exception as e:
            print(f"  [IntegratedGrad] HATA: {e}")

        # --- GNNExplainer ---
        # GNNExplainer kendi icinde tahmin yapip label'i otomatik belirler;
        # disaridan label almaz.
        try:
            exp_gnn = gnn_exp.get_explanation_graph(
                x=x,
                edge_index=edge_index,
                forward_kwargs=forward_kwargs,
            )
            print_node_importance(exp_gnn, processed, "GNNExplainer")
            graph_results["gnn_explainer"] = exp_gnn
        except Exception as e:
            print(f"  [GNNExplainer] HATA: {e}")

        print()
        results.append(graph_results)
        processed += 1

    # ------------------------------------------------------------------
    # 5. Ozet
    # ------------------------------------------------------------------
    print("=== Entegrasyon tamamlandi ===")
    print(f"    Toplam aciklanan grafik: {len(results)}")

    successful = {name: 0 for name in ("grad", "integrated_grad", "gnn_explainer")}
    for r in results:
        for name in successful:
            if name in r:
                successful[name] += 1

    for name, count in successful.items():
        print(f"    {name:<22}: {count}/{len(results)} basarili")

    return results


# ---------------------------------------------------------------------------
# Ek yardimci: tek bir grafigi interaktif olarak acikla
# ---------------------------------------------------------------------------

def explain_single_graph(
    gnn_nets: GnnNets,
    data,
    explainer_name: str = "grad",
) -> "Explanation":
    """
    Tek bir grafigi belirtilen GraphXAI aciklayicisiyla aciklar.

    Args:
        gnn_nets: Egitilmis GnnNets modeli.
        data: torch_geometric.data.Data objesi (tek grafik).
        explainer_name: 'grad', 'integrated_grad' veya 'gnn_explainer'.

    Returns:
        GraphXAI Explanation nesnesi.
    """
    if not GRAPHXAI_AVAILABLE:
        raise RuntimeError("GraphXAI yuklu degil.")

    wrapper = ProtGNNWrapper(gnn_nets)
    wrapper.eval()

    criterion = nn.CrossEntropyLoss()
    x          = data.x.to(model_args.device)
    edge_index = data.edge_index.to(model_args.device)
    label      = data.y.to(model_args.device)
    null_batch = torch.zeros(x.size(0), dtype=torch.long, device=model_args.device)
    forward_kwargs = {"batch": null_batch}

    if explainer_name == "grad":
        explainer = GradExplainer(wrapper, criterion=criterion)
        return explainer.get_explanation_graph(
            x=x, edge_index=edge_index, label=label, forward_kwargs=forward_kwargs
        )
    elif explainer_name == "integrated_grad":
        explainer = IntegratedGradExplainer(wrapper, criterion=criterion)
        return explainer.get_explanation_graph(
            x=x, edge_index=edge_index, label=label, forward_kwargs=forward_kwargs
        )
    elif explainer_name == "gnn_explainer":
        from graphxai.explainers.gnn_explainer import GNNExplainer as _GNNExp
        explainer = _GNNExp(wrapper)
        return explainer.get_explanation_graph(
            x=x, edge_index=edge_index, forward_kwargs=forward_kwargs
        )
    else:
        raise ValueError(f"Bilinmeyen aciklayici: {explainer_name!r}. "
                         "Secenekler: 'grad', 'integrated_grad', 'gnn_explainer'")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_graphxai_integration(num_test_graphs=5)
