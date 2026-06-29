"""
Generate cautious clinical-language explanations from enriched GraphXAI outputs.

Run:
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/generate_clinical_explanations.py --limit 5
"""

import argparse
import csv
import glob
import json
import os
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from protgnn_analysis.config import DATA_DIR, OUTPUTS_DIR

DEFAULT_EXPLAINER = "IntegratedGradExplainer"
LABELS = {0: "HOME", 1: "ADMITTED"}
VITAL_RULES = {
    "heartrate": ("elevated heart rate", lambda v: v > 100),
    "vs_heartrate": ("elevated heart rate", lambda v: v > 100),
    "o2sat": ("lower oxygen saturation", lambda v: v < 95),
    "vs_o2sat": ("lower oxygen saturation", lambda v: v < 95),
    "resprate": ("elevated respiratory rate", lambda v: v > 20),
    "vs_resprate": ("elevated respiratory rate", lambda v: v > 20),
    "sbp": ("abnormal systolic blood pressure", lambda v: v < 90 or v > 180),
    "vs_sbp": ("abnormal systolic blood pressure", lambda v: v < 90 or v > 180),
    "dbp": ("abnormal diastolic blood pressure", lambda v: v < 60 or v > 120),
    "vs_dbp": ("abnormal diastolic blood pressure", lambda v: v < 60 or v > 120),
    "temperature": ("abnormal temperature", lambda v: v < 36 or v > 38),
    "vs_temperature": ("abnormal temperature", lambda v: v < 36 or v > 38),
    "acuity": ("higher triage acuity", lambda v: v >= 3),
}


def _label_name(value):
    return LABELS.get(int(value), str(value))


def _load_metadata(results_dir):
    metadata_path = results_dir / "dataset_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(
            "Clinical explanation metadata is missing. Rerun "
            "`python scripts/train_and_explain.py --no_prot --explain_n 5` "
            "so dataset_metadata.json and enriched graph JSON files are created."
        )
    return json.loads(metadata_path.read_text())


def _load_graph(path):
    graph = json.loads(path.read_text())
    required = ["patient_index", "node_patient_indices", "feature_names", "explanations"]
    missing = [field for field in required if field not in graph or graph[field] is None]
    if missing:
        raise SystemExit(
            f"{path} is missing clinical metadata fields {missing}. "
            "Rerun train_and_explain to regenerate enriched explanation JSON files."
        )
    return graph


def _feature_signal(feature_name, value):
    if pd.isna(value):
        return None
    if feature_name in VITAL_RULES:
        phrase, predicate = VITAL_RULES[feature_name]
        try:
            if predicate(float(value)):
                return phrase
        except (TypeError, ValueError):
            return None
    if feature_name.startswith("med_") and float(value) > 0:
        med = feature_name[4:].replace("_", " ")
        return f"{med} medication indicator"
    if feature_name.startswith("transport_") and float(value) > 0:
        transport = feature_name.split("_", 1)[1].lower()
        return f"{transport} arrival transport indicator"
    if feature_name.startswith("gender_") and float(value) > 0:
        return f"{feature_name.replace('_', ' ').lower()} indicator"
    if feature_name.startswith("race_") and float(value) > 0:
        return f"{feature_name.replace('_', ' ').lower()} indicator"
    return None


def _top_feature_entries(graph, explainer):
    exp = graph["explanations"].get(explainer, {})
    if "error" in exp:
        raise SystemExit(f"{explainer} failed for graph {graph['graph_idx']}: {exp['error']}")
    top = exp.get("feature_importance", {}).get("top_features")
    if not top:
        raise SystemExit(
            f"{explainer} feature importance is missing for graph {graph['graph_idx']}. "
            "Rerun train_and_explain after the clinical explanation changes."
        )
    return top


def _top_node_entries(graph, explainer):
    exp = graph["explanations"].get(explainer, {})
    return exp.get("top_nodes") or []


def _clinical_signals(row, top_features, max_items=6):
    signals = []
    seen = set()
    for feature in top_features:
        name = feature.get("name")
        if name not in row:
            continue
        phrase = _feature_signal(name, row[name])
        if phrase and phrase not in seen:
            signals.append({
                "feature": name,
                "phrase": phrase,
                "value": row[name],
                "score": feature.get("score", 0.0),
            })
            seen.add(phrase)
        if len(signals) >= max_items:
            break
    return signals


def _important_neighbors(graph, max_neighbors=3):
    node_patient_indices = graph["node_patient_indices"]
    neighbors = []
    for node in _top_node_entries(graph, DEFAULT_EXPLAINER):
        node_idx = node["index"]
        if node_idx == 0 or node_idx >= len(node_patient_indices):
            continue
        neighbors.append({
            "node_index": node_idx,
            "patient_index": int(node_patient_indices[node_idx]),
            "score": node["score"],
        })
        if len(neighbors) >= max_neighbors:
            break
    return neighbors


def _neighbor_patterns(df, neighbors, top_features, max_items=4):
    if not neighbors:
        return []
    patterns = []
    seen = set()
    for feature in top_features:
        name = feature.get("name")
        if name not in df.columns:
            continue
        present = 0
        for neighbor in neighbors:
            phrase = _feature_signal(name, df.iloc[neighbor["patient_index"]][name])
            if phrase:
                present += 1
        if present > 0:
            phrase = _feature_signal(name, df.iloc[neighbors[0]["patient_index"]][name])
            if phrase and phrase not in seen:
                patterns.append(f"{phrase} in {present}/{len(neighbors)} influential similar patients")
                seen.add(phrase)
        if len(patterns) >= max_items:
            break
    return patterns


def _sentence_list(items):
    if not items:
        return "no clearly abnormal high-attribution clinical signals"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _format_value(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return str(value)


def _plain_phrase(text):
    replacements = {
        " arrival transport indicator": " arrival",
        " medication indicator": "",
        " indicator": "",
        "gender f": "female",
        "gender m": "male",
        "race white": "white race",
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _compact_signal(signal):
    return f"{_plain_phrase(signal['phrase'])} ({_format_value(signal['value'])})"


def _signal_details(signals):
    if not signals:
        return "None"
    return " | ".join(_compact_signal(signal) for signal in signals[:4])


def _neighbor_details(neighbors):
    if not neighbors:
        return "None"
    return " | ".join(
        f"patient {item['patient_index']}"
        for item in neighbors
    )


def _pattern_details(patterns):
    if not patterns:
        return "None"
    return " | ".join(_plain_phrase(pattern) for pattern in patterns[:2])


def _prototype_summary_text(graph):
    evidence = graph.get("prototype_evidence", {})
    if not evidence.get("enabled"):
        return "Disabled"
    if not evidence.get("available"):
        return "Unavailable"

    closest = evidence.get("top_closest", [])
    contributors = evidence.get("top_contributors", [])
    parts = {"nearest": "", "support": "", "against": ""}
    if closest:
        item = closest[0]
        parts["nearest"] = (
            f"nearest P{item['prototype_index']} "
            f"{_label_name(item['prototype_class'])} d={item['distance']:.2f}"
        )
    if contributors:
        supporting = [item for item in contributors if item["contribution_to_pred"] > 0]
        opposing = [item for item in contributors if item["contribution_to_pred"] < 0]
        if supporting:
            item = supporting[0]
            parts["support"] = (
                f"supports: P{item['prototype_index']} "
                f"{_label_name(item['prototype_class'])} {item['contribution_to_pred']:.3f}"
            )
        if opposing:
            item = opposing[0]
            parts["against"] = (
                f"opposes: P{item['prototype_index']} "
                f"{_label_name(item['prototype_class'])} {item['contribution_to_pred']:.3f}"
            )
    return " | ".join(value for value in parts.values() if value) or "None"


def _prototype_columns(graph):
    evidence = graph.get("prototype_evidence", {})
    columns = {
        "nearest_prototype": "",
        "nearest_prototype_class": "",
        "nearest_prototype_distance": "",
        "supporting_prototype": "",
        "supporting_prototype_class": "",
        "supporting_prototype_contribution": "",
        "opposing_prototype": "",
        "opposing_prototype_class": "",
        "opposing_prototype_contribution": "",
    }
    if not evidence.get("available"):
        return columns

    closest = evidence.get("top_closest", [])
    contributors = evidence.get("top_contributors", [])
    if closest:
        item = closest[0]
        columns["nearest_prototype"] = f"P{item['prototype_index']}"
        columns["nearest_prototype_class"] = _label_name(item["prototype_class"])
        columns["nearest_prototype_distance"] = round(float(item["distance"]), 3)

    supporting = [item for item in contributors if item["contribution_to_pred"] > 0]
    opposing = [item for item in contributors if item["contribution_to_pred"] < 0]
    if supporting:
        item = supporting[0]
        columns["supporting_prototype"] = f"P{item['prototype_index']}"
        columns["supporting_prototype_class"] = _label_name(item["prototype_class"])
        columns["supporting_prototype_contribution"] = round(float(item["contribution_to_pred"]), 3)
    if opposing:
        item = opposing[0]
        columns["opposing_prototype"] = f"P{item['prototype_index']}"
        columns["opposing_prototype_class"] = _label_name(item["prototype_class"])
        columns["opposing_prototype_contribution"] = round(float(item["contribution_to_pred"]), 3)
    return columns


def _supporting_prototype_phrase(graph):
    evidence = graph.get("prototype_evidence", {})
    if not evidence.get("available"):
        return None
    contributors = evidence.get("top_contributors", [])
    supporting = [item for item in contributors if item["contribution_to_pred"] > 0]
    if not supporting:
        return None
    item = supporting[0]
    return (
        f"prototype {item['prototype_index']} "
        f"({_label_name(item['prototype_class'])}) supported the prediction"
    )


def _decision_reason(pred, signals, patterns, graph):
    signal_text = ", ".join(_plain_phrase(signal["phrase"]) for signal in signals[:3]) or "no clear patient signal"
    pieces = [f"{pred}: {signal_text}"]
    if patterns:
        pieces.append("similar patients: " + _pattern_details(patterns))

    prototype_phrase = _supporting_prototype_phrase(graph)
    if prototype_phrase:
        pieces.append(_plain_phrase(prototype_phrase))
    return " | ".join(pieces)


def _graph_sort_key(path):
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return path.stem


def _prototype_lines(graph):
    evidence = graph.get("prototype_evidence", {})
    if not evidence.get("enabled"):
        return [
            "Prototype evidence",
            "------------------",
            "Prototype learning was disabled for this run.",
        ], {
            "prototype_enabled": False,
            "closest_prototype": "",
            "closest_prototype_class": "",
            "closest_prototype_distance": "",
            "top_prototype_contributors": "",
        }

    if not evidence.get("available"):
        return [
            "Prototype evidence",
            "------------------",
            "Prototype learning was enabled, but prototype distances were not available in this explanation file.",
        ], {
            "prototype_enabled": True,
            "closest_prototype": "",
            "closest_prototype_class": "",
            "closest_prototype_distance": "",
            "top_prototype_contributors": "",
        }

    closest = evidence.get("top_closest", [])
    contributors = evidence.get("top_contributors", [])
    lines = [
        "Prototype evidence",
        "------------------",
    ]

    if closest:
        item = closest[0]
        lines.append(
            "The nearest learned prototype was "
            f"prototype {item['prototype_index']} "
            f"({_label_name(item['prototype_class'])} prototype, "
            f"distance {item['distance']:.4f})."
        )
    else:
        lines.append("No nearest prototype was recorded.")

    if contributors:
        desc = ", ".join(
            f"prototype {p['prototype_index']} ({_label_name(p['prototype_class'])}, "
            f"contribution {p['contribution_to_pred']:.4f})"
            for p in contributors[:3]
        )
        lines.append(f"The strongest prototype-level contributors to the prediction were {desc}.")

    lines.append(
        "This prototype evidence describes similarity in the model embedding space, "
        "not a direct clinical diagnosis match."
    )

    return lines, {
        "prototype_enabled": True,
        "closest_prototype": closest[0]["prototype_index"] if closest else "",
        "closest_prototype_class": _label_name(closest[0]["prototype_class"]) if closest else "",
        "closest_prototype_distance": round(float(closest[0]["distance"]), 6) if closest else "",
        "top_prototype_contributors": "; ".join(
            f"{p['prototype_index']}:{_label_name(p['prototype_class'])}:{p['contribution_to_pred']:.6f}"
            for p in contributors[:3]
        ),
    }


def _render_explanation(graph, df, explainer):
    patient_index = int(graph["patient_index"])
    patient_row = df.iloc[patient_index]
    top_features = _top_feature_entries(graph, explainer)
    signals = _clinical_signals(patient_row, top_features)
    signal_phrases = [s["phrase"] for s in signals]
    neighbors = _important_neighbors(graph)
    patterns = _neighbor_patterns(df, neighbors, top_features)

    pred = LABELS.get(int(graph["pred_label"]), str(graph["pred_label"]))
    truth = LABELS.get(int(graph["true_label"]), str(graph["true_label"]))
    confidence_word = "correctly" if graph["correct"] else "incorrectly"
    prototype_section, prototype_summary = _prototype_lines(graph)
    signal_values = [_compact_signal(signal) for signal in signals[:4]]
    signal_values += [""] * (4 - len(signal_values))
    neighbor_values = [str(item["patient_index"]) for item in neighbors[:3]]
    neighbor_values += [""] * (3 - len(neighbor_values))
    pattern_values = [_plain_phrase(pattern) for pattern in patterns[:2]]
    pattern_values += [""] * (2 - len(pattern_values))

    lines = [
        f"Graph {graph['graph_idx']} Clinical Explanation",
        "=" * 40,
        f"Prediction: {pred}",
        f"True label: {truth}",
        f"Model outcome: {confidence_word} classified",
        f"Patient row index: {patient_index}",
        "",
        "Patient-level signals",
        "---------------------",
        f"The model emphasized {_sentence_list(signal_phrases)}.",
    ]

    if signals:
        lines.append("")
        lines.append("Highest-attribution clinical features:")
        for signal in signals:
            value = signal["value"]
            if isinstance(value, (float, np.floating)):
                value = round(float(value), 4)
            lines.append(f"- {signal['feature']}: {value} ({signal['phrase']})")

    lines.extend([
        "",
        "Similar-patient context",
        "-----------------------",
    ])
    if neighbors:
        neighbor_desc = ", ".join(
            f"node {n['node_index']} -> patient row {n['patient_index']}" for n in neighbors
        )
        lines.append(f"The most influential similar patients were {neighbor_desc}.")
    else:
        lines.append("No similar-patient nodes were strongly emphasized by the selected explainer.")

    if patterns:
        lines.append("Among those similar patients, the model-highlighted patterns included:")
        for pattern in patterns:
            lines.append(f"- {pattern}")

    lines.extend(["", *prototype_section])

    lines.extend([
        "",
        "Interpretation",
        "--------------",
        (
            "This suggests the model associated the prediction with the highlighted "
            "clinical profile and its similar-patient neighborhood."
        ),
        (
            "The explanation should not be interpreted as causal clinical evidence "
            "or as a standalone medical decision rule."
        ),
        "",
    ])
    raw_row = {
        "graph_idx": graph["graph_idx"],
        "patient_index": patient_index,
        "prediction": pred,
        "true_label": truth,
        "correct": graph["correct"],
        "top_signals": "; ".join(signal_phrases),
        "important_neighbors": "; ".join(str(n["patient_index"]) for n in neighbors),
        "neighbor_patterns": "; ".join(patterns),
        **prototype_summary,
    }
    readable_row = {
        "graph": graph["graph_idx"],
        "patient_row": patient_index,
        "prediction": pred,
        "actual": truth,
        "result": "correct" if graph["correct"] else "wrong",
        "patient_signal_1": signal_values[0],
        "patient_signal_2": signal_values[1],
        "patient_signal_3": signal_values[2],
        "patient_signal_4": signal_values[3],
        "similar_patient_1": neighbor_values[0],
        "similar_patient_2": neighbor_values[1],
        "similar_patient_3": neighbor_values[2],
        "similar_pattern_1": pattern_values[0],
        "similar_pattern_2": pattern_values[1],
        **_prototype_columns(graph),
        "one_line_reason": _decision_reason(pred, signals, patterns, graph),
    }
    return "\n".join(lines), raw_row, readable_row


def _write_csv(path, rows, delimiter=","):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _write_excel(path, rows):
    try:
        pd.DataFrame(rows).to_excel(path, index=False)
        return
    except Exception:
        pass

    headers = list(rows[0].keys())

    def col_name(index):
        name = ""
        index += 1
        while index:
            index, rem = divmod(index - 1, 26)
            name = chr(65 + rem) + name
        return name

    def cell_xml(row_idx, col_idx, value):
        ref = f"{col_name(col_idx)}{row_idx}"
        if value == "":
            return f'<c r="{ref}"/>'
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return f'<c r="{ref}"><v>{value}</v></c>'
        text = escape(str(value))
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    sheet_rows = []
    sheet_rows.append(
        '<row r="1">' + "".join(cell_xml(1, idx, header) for idx, header in enumerate(headers)) + "</row>"
    )
    for row_idx, row in enumerate(rows, start=2):
        sheet_rows.append(
            f'<row r="{row_idx}">'
            + "".join(cell_xml(row_idx, col_idx, row.get(header, "")) for col_idx, header in enumerate(headers))
            + "</row>"
        )

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetData>'
        + "".join(sheet_rows)
        + "</sheetData></worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _markdown_table(rows):
    lines = ["# Patient decision explanations", ""]
    for row in rows:
        verdict = "OK" if row["result"] == "correct" else "WRONG"
        patient_evidence = " | ".join(
            value for value in (
                row["patient_signal_1"],
                row["patient_signal_2"],
                row["patient_signal_3"],
                row["patient_signal_4"],
            )
            if value
        ) or "None"
        similar_patterns = " | ".join(
            value for value in (row["similar_pattern_1"], row["similar_pattern_2"])
            if value
        ) or "None"
        prototype = (
            f"nearest {row['nearest_prototype']} {row['nearest_prototype_class']} "
            f"d={row['nearest_prototype_distance']}; "
            f"supports {row['supporting_prototype']} {row['supporting_prototype_class']} "
            f"{row['supporting_prototype_contribution']}; "
            f"opposes {row['opposing_prototype']} {row['opposing_prototype_class']} "
            f"{row['opposing_prototype_contribution']}"
        )
        lines.extend([
            f"## Graph {row['graph']} - patient {row['patient_row']} - {verdict}",
            "",
            f"Prediction: **{row['prediction']}**. Actual: **{row['actual']}**.",
            "",
            f"- Patient evidence: {patient_evidence}",
            f"- Similar-patient pattern: {similar_patterns}",
            f"- Prototype: {prototype}",
            "",
        ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate clinical explanations from GraphXAI outputs")
    parser.add_argument("--results_dir", default=str(Path(OUTPUTS_DIR) / "results"))
    parser.add_argument("--csv", default=str(Path(DATA_DIR) / "merged_ed_no_icd_sample_20k.csv"))
    parser.add_argument("--explainer", default=DEFAULT_EXPLAINER)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    explanations_dir = results_dir / "explanations"
    output_dir = results_dir / "clinical_explanations"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_metadata(results_dir)
    # Use the trained dataset's label_mapping (e.g. 30 disease names) instead of
    # the binary HOME/ADMITTED default, so multi-class predictions are named.
    _lm = (meta or {}).get("label_mapping") or {}
    if _lm:
        LABELS.clear()
        LABELS.update({int(k): v for k, v in _lm.items()})
    df = pd.read_csv(args.csv)
    graph_paths = sorted(
        (Path(p) for p in glob.glob(str(explanations_dir / "graph_*.json"))),
        key=_graph_sort_key,
    )
    if args.limit is not None:
        graph_paths = graph_paths[:args.limit]
    if not graph_paths:
        raise SystemExit(f"No graph explanation JSON files found in {explanations_dir}")

    raw_rows = []
    readable_rows = []
    report_sections = []
    for graph_path in graph_paths:
        graph = _load_graph(graph_path)
        text, raw_row, readable_row = _render_explanation(graph, df, args.explainer)
        (output_dir / f"graph_{graph['graph_idx']}.txt").write_text(text)
        raw_rows.append(raw_row)
        readable_rows.append(readable_row)
        report_sections.append(text)

    _write_csv(output_dir / "summary.csv", readable_rows, delimiter=";")
    _write_csv(output_dir / "summary_comma.csv", readable_rows)
    _write_csv(output_dir / "summary_raw.csv", raw_rows)
    _write_excel(output_dir / "summary.xlsx", readable_rows)
    (output_dir / "summary.md").write_text(_markdown_table(readable_rows))

    (output_dir / "clinical_report.txt").write_text("\n\n".join(report_sections))
    print(f"Wrote {len(readable_rows)} clinical explanations to {output_dir}")


if __name__ == "__main__":
    main()
