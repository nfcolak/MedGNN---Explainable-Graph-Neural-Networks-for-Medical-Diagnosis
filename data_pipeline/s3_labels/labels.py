"""Bind visit-level targets to the event-graph artifact without changing graph JSONL.

The event graph production artifact intentionally contains no labels. This module
creates a separate, immutable metadata binding: raw ``diagnosis.csv`` rows are
resolved by the exact ``stay_id`` and mapped to the existing 30-class
``disease_1`` ordering. Subject/split checks remain explicit and no target is
recovered from subject order or visit ordinal.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_ROOT = REPO / "comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2"
DEFAULT_RAW_ROOT = REPO / "data/Original CSVs"
DEFAULT_LABELS = REPO / "comparison/standardized/native_inputs/protgsat_snapshot_v1/contract.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o600)


def _load_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    labels = data.get("labels") or data.get("reference_class_order_only")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError("An explicit ordered label list is required")
    if len(set(labels)) != len(labels):
        raise ValueError("Label order contains duplicates")
    return labels


def _load_disease_merges() -> dict[str, str]:
    """Load the existing merge policy without executing the data-prep script."""
    path = REPO / "shared/data_prep/merge_ed.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if "DISEASE_MERGES" in names:
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                raise ValueError("DISEASE_MERGES is not a string mapping")
            return value
    raise ValueError("DISEASE_MERGES was not found in merge_ed.py")


def _canonical_diagnoses(raw_root: Path) -> pd.DataFrame:
    diagnosis_path = raw_root / "diagnosis.csv"
    mapping_path = raw_root / "icd9_to_icd10_mapping.csv"
    diagnoses = pd.read_csv(diagnosis_path)
    required = {"subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"}
    if not required.issubset(diagnoses.columns):
        raise ValueError(f"diagnosis.csv missing columns: {sorted(required - set(diagnoses.columns))}")
    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.strip().str.upper()
    diagnoses["icd_title"] = diagnoses["icd_title"].astype(str).str.strip()
    diagnoses["stay_id"] = diagnoses["stay_id"].astype(str).str.strip()
    diagnoses["seq_num"] = pd.to_numeric(diagnoses["seq_num"], errors="coerce")

    icd10_titles = (
        diagnoses[diagnoses["icd_version"].astype(int) == 10]
        .dropna(subset=["icd_title"])
        .drop_duplicates("icd_code")
        .set_index("icd_code")["icd_title"]
        .to_dict()
    )
    mapping = pd.read_csv(mapping_path)
    required_mapping = {"no_map", "approximate", "icd9_code", "icd10_code"}
    if not required_mapping.issubset(mapping.columns):
        raise ValueError("icd9_to_icd10_mapping.csv has an unexpected schema")
    icd9_to_icd10 = (
        mapping[mapping["no_map"] == 0]
        .sort_values("approximate")
        .drop_duplicates("icd9_code", keep="first")
        .assign(
            icd9_code=lambda frame: frame["icd9_code"].astype(str).str.upper(),
            icd10_code=lambda frame: frame["icd10_code"].astype(str).str.upper(),
        )
        .set_index("icd9_code")["icd10_code"]
        .to_dict()
    )

    def icd10_code(row: Any) -> str:
        code = row.icd_code
        return code if int(row.icd_version) == 10 else icd9_to_icd10.get(code, "")

    diagnoses["icd10_code"] = [icd10_code(row) for row in diagnoses.itertuples()]
    diagnoses["canon_title"] = [
        icd10_titles.get(code, "").split(",")[0].strip()
        for code in diagnoses["icd10_code"]
    ]
    diagnoses["cat3"] = diagnoses["icd10_code"].str[:3]
    first = diagnoses["cat3"].str[0]
    excluded = first.isin(["V", "W", "X", "Y", "Z"]) | diagnoses["cat3"].eq("") | diagnoses["cat3"].isna()
    symptom = (first == "R") & ~excluded
    disease = ~excluded & ~symptom & diagnoses["canon_title"].ne("") & diagnoses["canon_title"].ne("nan")

    clean = diagnoses[disease]
    category_name = clean.groupby("cat3")["canon_title"].agg(lambda values: values.value_counts().index[0]).to_dict()
    category_name = {key: value for key, value in category_name.items() if value}
    named = disease & diagnoses["cat3"].isin(category_name)
    category_patients = diagnoses.loc[named].drop_duplicates(["stay_id", "cat3"])["cat3"].value_counts()
    merges = _load_disease_merges()

    def final_label(category: str) -> str:
        title = category_name.get(category, "")
        return (merges.get(title) or title) if title else ""

    ranked_labels: list[str] = []
    for category in category_patients.index:
        label = final_label(category)
        if label and label not in ranked_labels:
            ranked_labels.append(label)
    # The authoritative benchmark keeps the first 30 final classes, then sorts
    # the resulting names in its shared loader. The caller supplies that order.
    kept = set(ranked_labels[:30])
    diagnoses["final_label"] = diagnoses["cat3"].map(category_name).map(
        lambda title: merges.get(title, title) if isinstance(title, str) else ""
    )
    return diagnoses[disease & diagnoses["cat3"].isin(
        {category for category in category_patients.index if final_label(category) in kept}
    )].copy()


def derive_stay_targets(raw_root: Path, labels: list[str]) -> pd.DataFrame:
    """Return one target per stay only when the final disease label is unambiguous."""
    diagnoses = _canonical_diagnoses(raw_root)
    ordered = diagnoses.sort_values(["stay_id", "seq_num"], kind="stable")
    rows: list[tuple[str, str, int]] = []
    for stay_id, group in ordered.groupby("stay_id", sort=False):
        unique_labels = list(dict.fromkeys(label for label in group["final_label"].tolist() if label in labels))
        if len(unique_labels) == 1:
            label = unique_labels[0]
            rows.append((str(stay_id), label, labels.index(label)))
    result = pd.DataFrame(rows, columns=["stay_id", "label", "target"])
    if result.empty:
        raise ValueError("No unambiguous stay-level targets were derived")
    if result["stay_id"].duplicated().any():
        raise ValueError("Duplicate stay-level target")
    return result


def build_binding(
    graph_root: Path = DEFAULT_GRAPH_ROOT,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output: Path | None = None,
    labels_path: Path = DEFAULT_LABELS,
) -> dict[str, Any]:
    """Create a target binding beside, not inside, the immutable graph artifact."""
    graph_root = Path(graph_root).resolve()
    raw_root = Path(raw_root).resolve()
    labels_path = Path(labels_path).resolve()
    output = (graph_root.parent / "first_recorded_lab_all_visits_v2_targets_v1") if output is None else Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite target binding: {output}")

    manifest = json.loads((graph_root / "manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("schema_version") != "event_graph_v1":
        raise ValueError("Only a completed event_graph_v1 artifact may be labelled")
    labels = _load_labels(labels_path)
    reference_labels = manifest.get("reference_class_order_only")
    if reference_labels != labels:
        raise ValueError("Label order differs from the event artifact's recorded reference order")

    cohort_path = graph_root / "cohort.csv"
    cohort = pd.read_csv(cohort_path, dtype={"sample_id": str, "subject_id": str, "stay_id": str})
    required = {"sample_id", "subject_id", "stay_id", "split"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"Cohort missing columns: {sorted(required - set(cohort.columns))}")
    if cohort["sample_id"].duplicated().any() or cohort["stay_id"].duplicated().any():
        raise ValueError("Cohort sample/stay identity is not unique")
    if set(cohort["split"]) - {"train", "validation", "test"}:
        raise ValueError("Unexpected split value")

    targets = derive_stay_targets(raw_root, labels)
    bound = cohort.merge(targets, on="stay_id", how="left", validate="one_to_one")
    bound["target"] = bound["target"].fillna(-1).astype("int64")
    bound["label"] = bound["label"].fillna("")
    if not bool(bound.loc[bound["target"] >= 0, "label"].map(lambda value: value in labels).all()):
        raise ValueError("Target binding contains a label outside the ordered label list")
    if bound.loc[bound["target"] >= 0, "subject_id"].isna().any():
        raise ValueError("Target binding lost subject identity")

    matched = bound[bound["target"] >= 0]
    by_split = {
        split: {
            "samples": int((matched["split"] == split).sum()),
            "patients": int(matched.loc[matched["split"] == split, "subject_id"].nunique()),
        }
        for split in ("train", "validation", "test")
    }
    subject_split = bound.groupby("subject_id")["split"].nunique()
    if bool((subject_split > 1).any()):
        raise ValueError("A subject crosses event-graph folds")

    output.mkdir(parents=True)
    bound.to_csv(output / "targets.csv", index=False)
    (output / "targets.csv").chmod(0o600)
    summary = {
        "schema_version": "event_graph_target_binding_v1",
        "status": "completed",
        "graph_artifact": str(graph_root),
        "graph_manifest_sha256": sha256(graph_root / "manifest.json"),
        "graph_sha256": manifest["graphs_sha256"],
        "cohort_sha256": manifest["cohort_sha256"],
        "raw_diagnosis_sha256": sha256(raw_root / "diagnosis.csv"),
        "icd_mapping_sha256": sha256(raw_root / "icd9_to_icd10_mapping.csv"),
        "merge_policy_sha256": sha256(REPO / "shared/data_prep/merge_ed.py"),
        "labels_path": str(labels_path),
        "labels": labels,
        "target_policy": "Existing disease_1 policy reproduced from diagnosis.csv by exact stay_id; only one final disease label per stay retained.",
        "counts": {
            "cohort_samples": int(len(bound)),
            "matched_samples": int(len(matched)),
            "unmatched_samples": int((bound["target"] < 0).sum()),
            "cohort_patients": int(bound["subject_id"].nunique()),
            "matched_patients": int(matched["subject_id"].nunique()),
            "patients_with_multiple_targets": int(matched.groupby("subject_id")["target"].nunique().gt(1).sum()),
            "by_split": by_split,
            "class_counts": {str(index): int((matched["target"] == index).sum()) for index in range(len(labels))},
        },
        "limitations": [
            "Target is derived from the visit's diagnosis record and is not an early-visibility claim.",
            "The graph manifest remains unlabelled; this file is a separate supervised binding.",
            "Unmatched and multi-disease stays are retained in targets.csv with target=-1 and excluded by the training runner.",
            "The event artifact is temporal_clean=false and uses storetime as an availability proxy.",
        ],
        "artifact_files": {"targets.csv": None},
    }
    summary["artifact_files"]["targets.csv"] = sha256(output / "targets.csv")
    _json_save(output / "binding_manifest.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()
    print(json.dumps(build_binding(args.graph_root, args.raw_root, args.output, args.labels), indent=2))
