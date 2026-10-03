# CEI-GNN overview deck (3 slides) + clinical-graph asset

Aggregate-only, docs-only. No patient data; the diagram is schematic.

| File | What |
|---|---|
| `cei-overview.pptx` | Editable 3-slide deck (speaker notes carry provenance) |
| `cei-overview.pdf` | PDF export of the PPTX by Microsoft PowerPoint for Mac (Keynote export timed out) |
| `assets/clinical-graph.svg` / `.png` | Reusable "Shared multi-visit clinical graph" diagram with legend, direction caveat, "Schematic - not patient data" |
| `build_cei_overview.py` | Rebuilds the SVG and PPTX (`python3 build_cei_overview.py`, needs python-pptx). PNG: render the SVG in a browser; PDF: export the PPTX from PowerPoint/Keynote. |

## Provenance
- Graph relations/directions: `comparison/standardized/clinical_graph_v2/graph.py` (docstring + `_link`); inverse/symmetric pairs and `rev:*`: `.../tensorize.py` lines 34-44; forward default `train.py` ~213-219, ~1046-1050; core study forward `cei_v3_run.py` ~210. Bidirectional mode only had a bounded train/dev smoke.
- CEI decomposition: `.../methods/cei_gnn.py`, `cei_gnn_v2.py`, `cei_gnn_v3.py`.
- Numbers and claims: `comparison/standardized/clinical_graph_v2/methods/cei/docs/cei-v3-delivery-2026-10-01.md`, `comparison/standardized/clinical_graph_v2/methods/cei/docs/cei-v3-evidence-2026-10-01.json`.
- Caveats: temporal availability is partly assumed (`temporal_clean=false`); not claimed leakage-free. v3 benefit over v2 not demonstrated; no winner claim.
