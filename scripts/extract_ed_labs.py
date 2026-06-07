"""
One-time extraction of a common lab panel from the huge MIMIC-IV labevents.csv
(~17 GB) into a small per-ED-stay table:  data/ed_labs.csv

Why a separate step: labevents is far too large to re-read on every merge. This
reads it ONCE in chunks, keeps only (a) a curated common-lab panel and (b) ED
patients, windows each lab to the ED stay it falls inside ([intime, outtime]),
averages repeats, and writes one compact row per stay_id with lab_<name> values.

Labs are objective measurements of the patient (like vitals), windowed to the
ED visit — used under a "diagnosis decision-support" framing (labs are available
when the diagnosis is made). A handful (troponin) are near-confirmatory; note
that in the thesis.

Run once:
    python3 scripts/extract_ed_labs.py
Then merge_ed.py picks up data/ed_labs.csv automatically.
"""

import os
import pandas as pd

# This script lives in <root>/scripts/ but reads/writes data in <root>/data/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
RAW = os.path.join(BASE, "Original CSVs")

# Curated common ED lab panel: MIMIC-IV labevents itemid -> short name.
LAB_ITEMS = {
    50862: "albumin",      50868: "anion_gap",   50882: "bicarbonate",
    50885: "bilirubin",    50893: "calcium",     50902: "chloride",
    50912: "creatinine",   50931: "glucose",     50960: "magnesium",
    50970: "phosphate",    50971: "potassium",   50983: "sodium",
    51006: "bun",          51221: "hematocrit",  51222: "hemoglobin",
    51237: "inr",          51265: "platelet",    51275: "ptt",
    51301: "wbc",          50813: "lactate",     51003: "troponin_t",
    50861: "alt",          50878: "ast",         50863: "alk_phos",
    51250: "mcv",          51279: "rbc",         50902: "chloride",
}
KEEP_IDS = set(LAB_ITEMS)

# plausible value clamps (drop obvious garbage); keys are short names
CLAMP = {
    "glucose": (10, 2000), "creatinine": (0.1, 30), "bun": (1, 250),
    "sodium": (100, 200), "potassium": (1, 12), "chloride": (60, 160),
    "bicarbonate": (2, 60), "anion_gap": (0, 60), "calcium": (2, 20),
    "magnesium": (0.3, 10), "phosphate": (0.3, 20), "albumin": (0.5, 7),
    "bilirubin": (0, 60), "alt": (1, 10000), "ast": (1, 10000),
    "alk_phos": (5, 3000), "hematocrit": (5, 75), "hemoglobin": (2, 25),
    "wbc": (0, 200), "platelet": (1, 2000), "mcv": (50, 130),
    "rbc": (1, 10), "inr": (0.5, 20), "ptt": (10, 200),
    "lactate": (0.1, 40), "troponin_t": (0, 50),
}


def main():
    edstays = pd.read_csv(os.path.join(RAW, "edstays.csv"),
                          usecols=["subject_id", "stay_id", "intime", "outtime"])
    edstays["intime"] = pd.to_datetime(edstays["intime"])
    edstays["outtime"] = pd.to_datetime(edstays["outtime"])
    ed_subjects = set(edstays["subject_id"].unique())
    print(f"ED stays: {len(edstays)} | ED subjects: {len(ed_subjects)}")

    path = os.path.join(RAW, "labevents.csv")
    print(f"Streaming {path} in chunks (panel={len(KEEP_IDS)} labs)...")
    cols = ["subject_id", "itemid", "charttime", "valuenum", "flag"]
    kept = []
    n_rows = 0
    for i, chunk in enumerate(pd.read_csv(path, usecols=cols, chunksize=2_000_000,
                                          low_memory=False)):
        n_rows += len(chunk)
        c = chunk[chunk["itemid"].isin(KEEP_IDS)
                  & chunk["subject_id"].isin(ed_subjects)
                  & chunk["valuenum"].notna()]
        if len(c):
            kept.append(c)
        if (i + 1) % 5 == 0:
            print(f"  ...{n_rows/1e6:.0f}M rows scanned, {sum(map(len, kept))} lab rows kept")
    labs = pd.concat(kept, ignore_index=True)
    print(f"Scanned {n_rows/1e6:.0f}M rows → {len(labs)} panel-lab rows for ED patients")

    labs["charttime"] = pd.to_datetime(labs["charttime"])
    labs["lab"] = labs["itemid"].map(LAB_ITEMS)

    # window each lab to the ED stay it falls inside: join by subject, keep rows
    # with intime <= charttime <= outtime.
    merged = labs.merge(edstays, on="subject_id", how="inner")
    in_window = ((merged["charttime"] >= merged["intime"]) &
                 (merged["charttime"] <= merged["outtime"]))
    merged = merged[in_window]
    print(f"Labs inside an ED stay window: {len(merged)}")

    # clamp implausible values
    for name, (lo, hi) in CLAMP.items():
        m = merged["lab"] == name
        bad = m & ((merged["valuenum"] < lo) | (merged["valuenum"] > hi))
        merged.loc[bad, "valuenum"] = pd.NA
    merged = merged.dropna(subset=["valuenum"])

    # mean value per (stay, lab) → wide table  →  lab_<name>
    agg = (merged.groupby(["stay_id", "lab"])["valuenum"].mean()
                 .unstack("lab"))
    agg.columns = ["lab_" + c for c in agg.columns]
    agg = agg.round(2)

    # abnormal flag per (stay, lab): 1 if ANY measurement was flagged 'abnormal'
    # during the stay (labevents `flag` column). Dense, clinically-actionable
    # signal (a doctor reacts to an abnormal lab) → lab_<name>_abn.
    merged["abn"] = (merged["flag"].astype(str).str.lower() == "abnormal").astype("int8")
    abn = (merged.groupby(["stay_id", "lab"])["abn"].max().unstack("lab"))
    abn.columns = ["lab_" + c + "_abn" for c in abn.columns]

    agg = agg.join(abn).reset_index()
    print(f"Lab value columns: {sum(c.startswith('lab_') and not c.endswith('_abn') for c in agg.columns)} "
          f"| abnormal-flag columns: {sum(c.endswith('_abn') for c in agg.columns)}")

    out = os.path.join(RAW, "ed_labs.csv")   # alongside the other Original CSVs
    agg.to_csv(out, index=False)
    cover = agg["stay_id"].nunique() / len(edstays)
    print(f"\nSaved: {out}  shape={agg.shape}")
    print(f"  stays with >=1 lab: {agg['stay_id'].nunique()} ({cover:.1%} of ED stays)")
    print(f"  lab columns: {[c for c in agg.columns if c.startswith('lab_')]}")


if __name__ == "__main__":
    main()
