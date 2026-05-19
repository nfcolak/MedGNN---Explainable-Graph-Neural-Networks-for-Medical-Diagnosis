"""GraphXAI explainer exports.

Some explainers depend on optional packages that are not required by this
project. Keep those imports best-effort so Grad, IntegratedGrad, and
GNNExplainer can be used without installing every optional GraphXAI dependency.
"""

from .grad import GradExplainer
from .integrated_grad import IntegratedGradExplainer
from .gnn_explainer import GNNExplainer

try:
    from .subgraphx import SubgraphX
except Exception:
    SubgraphX = None

try:
    from .cam import CAM, GradCAM
except Exception:
    CAM = GradCAM = None

try:
    from .guidedbp import GuidedBP
except Exception:
    GuidedBP = None

try:
    from .gnn_lrp import GNN_LRP
except Exception:
    GNN_LRP = None

try:
    from .random import RandomExplainer
except Exception:
    RandomExplainer = None

try:
    from .graphlime import GraphLIME
except Exception:
    GraphLIME = None

try:
    from .pg_explainer import PGExplainer
except Exception:
    PGExplainer = None

try:
    from .pgm_explainer import PGMExplainer
except Exception:
    PGMExplainer = None

