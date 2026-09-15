"""Single source of truth for benchmark graph-structure policies.

The methods build graphs differently: ProtGNN and GSAT consume one PyG `Data`
per patient, while GraphCare builds a per-patient subgraph over a global KG.
What they share is the *edge policy*: given a patient's concept nodes, which
edges connect them. That policy is selected by name so each method can run the
same benchmark topology without editing code.

Structures use the patient hub plus method-eligible concept nodes. For the
standardized ``disease_1`` benchmark, all three methods use the conservative
common membership subset: patient hub, training-fitted medication nodes, and
training-fitted pre-diagnosis ``chiefcomplaint_*`` nodes. Diagnosis-derived
``symptom_*``, diagnosis/target inputs, and vital nodes are excluded. GraphCare
cannot consume the identical numeric vital measurements, so standardized
ProtGNN/GSAT do not create categorical vital-name-only substitutes. Legacy mode
retains its historical input policy:

    star      patient hub ↔ each concept only (baseline; no concept↔concept edges)
    cooccur   star + concept↔concept where population PMI > threshold (cross-type)
    ontology  star + concept↔concept sharing an ontology class
              (meds → therapeutic class, ICD/symptom → ICD chapter)
    full      record-local star + (cooccur ∪ ontology)
    full_kg_expanded
              GraphCare-only full topology plus record-external 1-hop KG nodes

Method flags say whether a method can realise that structure. The common
ordering (star ⊂ cooccur/ontology ⊂ full on edges) makes the set a clean
monotone ablation ladder. ``full_kg_expanded`` is exploratory and excluded from
the primary cross-method table.
"""
from collections import OrderedDict

SUPPORTED_METHODS = ("protgnn", "gsat", "graphcare")

# name -> metadata. OrderedDict provides stable CLI and report display.
GRAPH_STRUCTURES = OrderedDict([
    ("star", {
        "protgnn": True, "gsat": True, "graphcare": True,
        "primary": True, "record_local": True, "kg_expanded": False,
        "desc": "Patient hub ↔ each concept only (baseline, no concept↔concept edges).",
    }),
    ("cooccur", {
        "protgnn": True, "gsat": True, "graphcare": True,
        "primary": True, "record_local": True, "kg_expanded": False,
        "desc": "Star + concept↔concept where population PMI > threshold (cross-type).",
    }),
    ("ontology", {
        "protgnn": True, "gsat": True, "graphcare": True,
        "primary": False, "record_local": True, "kg_expanded": False,
        "desc": "Star + concept↔concept sharing an ontology class (drug class / ICD chapter).",
    }),
    ("full", {
        "protgnn": True, "gsat": True, "graphcare": True,
        "primary": False, "record_local": True, "kg_expanded": False,
        "desc": "Record-local star + cooccur ∪ ontology; no record-external nodes.",
    }),
    ("full_kg_expanded", {
        "protgnn": False, "gsat": False, "graphcare": True,
        "primary": False, "record_local": False, "kg_expanded": True,
        "desc": "GraphCare-only full topology plus record-external 1-hop KG neighbours.",
    }),
])

DEFAULT_STRUCTURE = "star"


def all_structures():
    return list(GRAPH_STRUCTURES.keys())


def all_methods():
    return list(SUPPORTED_METHODS)


def supported_structures(method):
    """Return structure names that the benchmark method can build."""
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
    if method not in SUPPORTED_METHODS:
        choices = ", ".join(repr(name) for name in SUPPORTED_METHODS)
        raise ValueError(f"method must be one of {choices}, got {method!r}")
