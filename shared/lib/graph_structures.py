"""Single source of truth for the GRAPH STRUCTURE (topology) used by ALL three
analyses (ProtGNN, GraphCare and GSAT).

The methods build graphs differently — ProtGNN and GSAT both emit one PyG `Data`
per patient (feature vectors on nodes; GSAT reuses ProtGNN's cache verbatim),
GraphCare builds a per-patient subgraph over a global KG (node ids + relation
ids). What they CAN share is the *edge policy*: given a patient's concept nodes,
which edges connect them. That policy is what we call the "graph structure", and
it is selected by name so the exact same pipeline runs for any topology without
editing code.

Structures (concept nodes = the patient's meds / labs / symptoms / chief
complaints / diagnoses; the PATIENT hub is always present so its demographic /
numeric features can propagate):

    star      patient hub ↔ each concept only (baseline; no concept↔concept edges)
    cooccur   star + concept↔concept where population PMI > threshold (cross-type)
    ontology  star + concept↔concept sharing an ontology class
              (meds → therapeutic class, ICD/symptom → ICD chapter)
    full      star + (cooccur ∪ ontology); GraphCare additionally expands to the
              patient's 1-hop KG neighbours that are NOT in the record

`protgnn` / `graphcare` / `gsat` flags say whether a method can realise that
structure. The ordering (star ⊂ cooccur/ontology ⊂ full on edges) makes the set
a clean monotone ablation ladder that all three methods realise, for a fair
comparison. GSAT consumes the same per-patient graphs as ProtGNN, so it supports
exactly the structures ProtGNN does.
"""
from collections import OrderedDict

# name -> metadata. `order` is only for stable display.
GRAPH_STRUCTURES = OrderedDict([
    ("star", {
        "protgnn": True, "graphcare": True, "gsat": True,
        "desc": "Patient hub ↔ each concept only (baseline, no concept↔concept edges).",
    }),
    ("cooccur", {
        "protgnn": True, "graphcare": True, "gsat": True,
        "desc": "Star + concept↔concept where population PMI > threshold (cross-type).",
    }),
    ("ontology", {
        "protgnn": True, "graphcare": True, "gsat": True,
        "desc": "Star + concept↔concept sharing an ontology class (drug class / ICD chapter).",
    }),
    ("full", {
        "protgnn": True, "graphcare": True, "gsat": True,
        "desc": "Star + cooccur ∪ ontology (GraphCare also expands to 1-hop KG neighbours).",
    }),
])

DEFAULT_STRUCTURE = "star"


def all_structures():
    return list(GRAPH_STRUCTURES.keys())


def supported_structures(method):
    """method: 'protgnn' | 'graphcare' | 'gsat' -> list of structure names it can build."""
    _check_method(method)
    return [name for name, meta in GRAPH_STRUCTURES.items() if meta[method]]


def is_supported(name, method):
    _check_method(method)
    return name in GRAPH_STRUCTURES and GRAPH_STRUCTURES[name][method]


def resolve(name, method):
    """Validate a structure name for a method; return the canonical name.

    Raises ValueError with an actionable message if unknown/unsupported.
    """
    _check_method(method)
    if name is None:
        return DEFAULT_STRUCTURE
    key = str(name).strip().lower()
    if key not in GRAPH_STRUCTURES:
        raise ValueError(
            f"Unknown graph structure '{name}'. "
            f"Choose one of: {', '.join(all_structures())}."
        )
    if not GRAPH_STRUCTURES[key][method]:
        raise ValueError(
            f"Graph structure '{key}' is not supported by {method}. "
            f"{method} supports: {', '.join(supported_structures(method))}."
        )
    return key


def graphs_root(data_dir):
    """The single main folder that collects every graph type's cache."""
    from pathlib import Path
    return Path(data_dir) / "graphs"


def structure_dir(data_dir, name, method):
    """<data_dir>/graphs/<structure>/<method>/  — one subfolder per graph type."""
    return graphs_root(data_dir) / name / method


def prompt_for_structure(method, default=DEFAULT_STRUCTURE, stream=None):
    """Interactively ask which graph structure to use (only supported ones shown).

    Returns the chosen name. Falls back to `default` on empty input or when
    stdin is not a TTY (non-interactive runs must pass --graph_structure).
    """
    import sys
    stream = stream or sys.stdout
    options = supported_structures(method)
    if default not in options:
        default = options[0]

    if not sys.stdin or not sys.stdin.isatty():
        print(f"[graph structure] non-interactive; defaulting to '{default}' "
              f"(pass --graph_structure to override).", file=stream)
        return default

    print(f"\nSelect graph structure for {method}:", file=stream)
    for i, name in enumerate(options, 1):
        mark = " (default)" if name == default else ""
        print(f"  {i}. {name}{mark} — {GRAPH_STRUCTURES[name]['desc']}", file=stream)
    while True:
        raw = input(f"Choice [1-{len(options)}] or name (Enter = {default}): ").strip()
        if not raw:
            return default
        if raw.lower() in options:
            return raw.lower()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  invalid — type 1-{len(options)} or one of: {', '.join(options)}")


def _check_method(method):
    if method not in ("protgnn", "graphcare", "gsat"):
        raise ValueError(f"method must be 'protgnn', 'graphcare' or 'gsat', got {method!r}")
