import pandas as pd
import numpy as np
import os
import re
import sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from med_standardizer import clean_med_list
from chiefcomplaint_standardizer import clean_complaint_list

# This script lives in <root>/scripts/ but reads/writes data in <root>/data/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
RAW_DIR  = os.path.join(BASE_DIR, "Original CSVs")

def load(name):
    return pd.read_csv(os.path.join(RAW_DIR, name))

edstays   = load("edstays.csv")
triage    = load("triage.csv")
vitalsign = load("vitalsign.csv")
diagnosis = load("diagnosis.csv")
medrecon  = load("medrecon.csv")
pyxis     = load("pyxis.csv")

# triage: one row per stay → merge directly
df = edstays.merge(triage, on=["subject_id", "stay_id"], how="left", suffixes=("", "_triage"))

# prior ED utilisation: total number of ED stays per patient (recurrent visitors
# tend to carry chronic disease). Leak-free — it's a count of visits, not the target.
_visit_counts = edstays.groupby("subject_id")["stay_id"].nunique()
df["n_ed_visits"] = df["subject_id"].map(_visit_counts).fillna(1).astype(int)

# age: from MIMIC-IV hosp patients.csv (anchor_age). 100% subject_id overlap with
# the ED cohort. Strong, leak-free demographic predictor (disease distribution is
# highly age-dependent: epilepsy vs Alzheimer vs alcohol intox vs UTI).
_patients = load("patients.csv")
_age_map = _patients.set_index("subject_id")["anchor_age"]
df["age"] = df["subject_id"].map(_age_map)
print(f"Age merged from patients.csv (missing: {df['age'].isna().sum()})")

# BMI: from hosp omr.csv. Use the most recent measurement ON/BEFORE the ED visit
# (leak-free as-of join). BMI is a stable body-habitus trait + known risk factor
# (obesity → diabetes/HTN). ~77% of ED patients have a BMI on record.
_omr = load("omr.csv")
_bmi = _omr[_omr["result_name"].str.contains("BMI", case=False, na=False)].copy()
_bmi["bmi"] = pd.to_numeric(_bmi["result_value"], errors="coerce")
_bmi = _bmi[(_bmi["bmi"] >= 10) & (_bmi["bmi"] <= 80)].copy()
_bmi["chartdate"] = pd.to_datetime(_bmi["chartdate"])
_bmi = _bmi[["subject_id", "chartdate", "bmi"]].dropna().sort_values("chartdate")
_stay_in = df[["subject_id", "stay_id", "intime"]].copy()
_stay_in["intime"] = pd.to_datetime(_stay_in["intime"])
_stay_in = _stay_in.dropna(subset=["intime"]).sort_values("intime")
_asof = pd.merge_asof(_stay_in, _bmi, left_on="intime", right_on="chartdate",
                      by="subject_id", direction="backward")
df["bmi"] = df["stay_id"].map(_asof.set_index("stay_id")["bmi"])
print(f"BMI merged from omr.csv (leak-free prior measurement, "
      f"coverage: {df['bmi'].notna().mean():.1%} of stays)")

# Lab panel from data/ed_labs.csv (produced once by extract_ed_labs.py from the
# 17GB labevents). Labs are objective measurements windowed to the ED stay —
# legitimate diagnostic input under a decision-support framing. Merged by stay_id.
_lab_path = os.path.join(RAW_DIR, "ed_labs.csv")
if os.path.exists(_lab_path):
    _labs = pd.read_csv(_lab_path)
    df = df.merge(_labs, on="stay_id", how="left")
    _lab_cols = [c for c in df.columns if c.startswith("lab_")]
    print(f"Merged {len(_lab_cols)} lab columns from ed_labs.csv "
          f"(coverage: {df[_lab_cols].notna().any(axis=1).mean():.1%} of stays)")
else:
    print("[WARN] ed_labs.csv not found — run `python3 scripts/extract_ed_labs.py` "
          "first to add labs (else continuing without labs)")

# vitalsign: multiple rows per stay → aggregate numerics.
# Convert temperature F→C up front so EVERY derived statistic is in Celsius, and
# clamp implausible raw values to NaN BEFORE aggregating so they cannot corrupt
# min/max/std. (The df-level VITAL_RANGES clamp below still guards triage vitals.)
vitalsign = vitalsign.copy()
vitalsign["temperature"] = (vitalsign["temperature"] - 32) * 5 / 9
_VS_RAW_RANGES = {"temperature": (25.0, 45.0), "heartrate": (5.0, 300.0),
                  "resprate": (1.0, 60.0), "o2sat": (50.0, 100.0),
                  "sbp": (40.0, 300.0), "dbp": (0.0, 200.0)}
for _c, (_lo, _hi) in _VS_RAW_RANGES.items():
    vitalsign.loc[(vitalsign[_c] < _lo) | (vitalsign[_c] > _hi), _c] = np.nan

# mean (as before) + min / max / std trend statistics per vital
_vs_vitals = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
_agg_spec = {"vs_count": ("charttime", "count")}
for _v in _vs_vitals:
    _agg_spec[f"vs_{_v}"] = (_v, "mean")
    _agg_spec[f"vs_{_v}_min"] = (_v, "min")
    _agg_spec[f"vs_{_v}_max"] = (_v, "max")
    _agg_spec[f"vs_{_v}_std"] = (_v, "std")
vs_agg = (
    vitalsign
    .groupby(["subject_id", "stay_id"])
    .agg(**_agg_spec)
    .reset_index()
)
# std is NaN for single-measurement stays → 0 variability
_std_cols = [c for c in vs_agg.columns if c.endswith("_std")]
vs_agg[_std_cols] = vs_agg[_std_cols].fillna(0.0)
df = df.merge(vs_agg, on=["subject_id", "stay_id"], how="left")

# diagnosis: multiple rows per stay.
# Split each visit's diagnoses (ordered by seq_num = clinician priority) into
#   - real DISEASE diagnoses  → disease_1 / disease_2 / disease_3   (max 3)
#   - SYMPTOM/sign diagnoses   → symptom_1 ... symptom_5            (max 5)
# stored as human-readable icd_title. External-cause (V/W/X/Y) and status (Z)
# codes are excluded from both. These columns are intended as a prediction
# TARGET (what is the patient's diagnosis), so they are kept out of the
# feature columns downstream.
# Vectorised classification of every diagnosis row into disease / symptom /
# excluded (per-group Python apply over ~400k groups was far too slow).
_d = diagnosis.copy()
_d["icd_code"] = _d["icd_code"].astype(str).str.strip().str.upper()
_d["icd_title"] = _d["icd_title"].astype(str).str.strip()

# --- Standardize diagnosis titles ----------------------------------------
# Problem: ICD-9 titles are ALL-CAPS abbreviations ("HYPOTENSION NOS") while
# ICD-10 titles are clean sentence-case ("Hypotension, unspecified"). Build a
# canonical code→title map from the ICD-10 rows (already clean) and re-assign
# every row's title from it, mapping ICD-9 codes through the ICD-9→ICD-10
# crosswalk first. Anything unmapped falls back to a Title-Cased version of the
# original so the output is uniformly cased.
_icd10_title = (
    _d[_d["icd_version"] == 10]
    .dropna(subset=["icd_title"])
    .drop_duplicates("icd_code")
    .set_index("icd_code")["icd_title"]
    .to_dict()
)
_icd9to10 = pd.read_csv(os.path.join(RAW_DIR, "icd9_to_icd10_mapping.csv"))
_icd9to10 = (
    _icd9to10[_icd9to10["no_map"] == 0]
    .sort_values("approximate")
    .drop_duplicates("icd9_code", keep="first")
    .assign(icd9_code=lambda d: d["icd9_code"].astype(str).str.upper(),
            icd10_code=lambda d: d["icd10_code"].astype(str).str.upper())
    .set_index("icd9_code")["icd10_code"]
    .to_dict()
)

def _canonical_title(code, version, original):
    # Return the canonical clean ICD-10 title, or "" if no canonical title
    # exists. Non-canonical diagnoses (mostly old ICD-9 abbreviations whose
    # meaning is ambiguous, e.g. "OTH SEQUELA, CHR LIV DIS") are dropped rather
    # than kept as messy fallback text — only ~8% of diagnosis rows and ~2% of
    # patients lose *all* diagnoses this way.
    code = str(code)
    if version == 10:
        return _icd10_title.get(code, "")
    mapped = _icd9to10.get(code)
    if mapped is not None and mapped in _icd10_title:
        return _icd10_title[mapped]
    return ""

def _trim_title(t):
    # Drop everything after the first comma: ICD-10 short titles append
    # non-informative modifiers ("..., unspecified", "..., initial encounter",
    # "..., not intractable") that explode the label space without clinical
    # value. "Urinary tract infection, site not specified" -> "Urinary tract
    # infection".
    if not t:
        return ""
    return t.split(",")[0].strip()

_d["icd_title"] = [
    _trim_title(_canonical_title(c, v, t))
    for c, v, t in zip(_d["icd_code"], _d["icd_version"], _d["icd_title"])
]

# Collapse every diagnosis to its 3-character ICD category (e.g. G43.109 -> G43,
# "Migraine"). Diagnoses must be standardized to ICD-10 codes first; ICD-9 codes
# are mapped via the crosswalk so the category is taken on the ICD-10 code.
def _to_icd10_code(code, version):
    code = str(code).upper()
    if version == 10:
        return code
    return _icd9to10.get(code, "")   # "" if unmappable

_d["icd10_code"] = [
    _to_icd10_code(c, v) for c, v in zip(_d["icd_code"], _d["icd_version"])
]
_d["cat3"] = _d["icd10_code"].str[:3]

_first = _d["cat3"].str[0]
# excluded: V/W/X/Y/Z (external cause / status) + empty (unmappable)
_excluded = _first.isin(["V", "W", "X", "Y", "Z"]) | _d["cat3"].eq("") | _d["cat3"].isna()
# symptom: ICD-10 R*
_symptom = (_first == "R") & ~_excluded
# disease: everything else with a usable (canonical) title
_disease = ~_excluded & ~_symptom & _d["icd_title"].ne("") & _d["icd_title"].ne("nan")

# --- Keep only the TOP-N most common disease categories ---
# Predicting the primary diagnosis is a hard multi-class task; restricting to the
# most frequent N categories keeps every class well-populated (>1000 patients)
# and the problem tractable. Categories without a valid ICD-10 title are dropped
# first so N is counted over named categories only.
# Number of FINAL clinical classes to keep (AFTER the DISEASE_MERGES collapse) —
# categories are selected below so that exactly this many merged classes remain.
# Env-overridable, e.g. for a broad all-label confusion study:
#   TOP_N_DISEASES=150 python3 scripts/merge_ed.py
# (a non-30 value suffixes the output filename so it never clobbers the main set).
_TOP_N_DISEASES = int(os.environ.get("TOP_N_DISEASES", 30))
_dis_rows = _d[_disease]

def _category_label(title_series):
    # Name a 3-char category by its single MOST FREQUENT canonical title.
    titles = [t for t in title_series if t and t != "nan"]
    if not titles:
        return ""
    return pd.Series(titles).value_counts().index[0]

_clean = _dis_rows[_dis_rows["icd_title"].ne("") & _dis_rows["icd_title"].ne("nan")]
_cat_name = (
    _clean.groupby("cat3")["icd_title"]
          .agg(_category_label)
          .to_dict()
)
_cat_name = {k: v for k, v in _cat_name.items() if v}   # drop unnamed/invalid
# rank named categories by distinct-patient count (selection done AFTER merges,
# below, so that TOP_N_DISEASES targets the number of FINAL clinical classes).
_named_rows = _dis_rows[_dis_rows["cat3"].isin(_cat_name)]
_cat_patients = _named_rows.drop_duplicates(["stay_id", "cat3"])["cat3"].value_counts()

# ── Merge clinically-equivalent disease classes ──────────────────────────────
# Confusion analysis (on the broad 150-class set) showed the model heavily — and
# clinically sensibly — confuses sub-types of the same clinical entity. These are
# collapsed into clinically-coherent classes. Mappings are explicit (raw ICD
# category titles) and curated to avoid wrong merges (e.g. throat abrasion is NOT
# a limb injury, a dental/diverticular abscess is NOT a skin infection).
DISEASE_MERGES = {
    # --- Diabetes (T1 + T2) ---
    "Type 1 diabetes mellitus without complications": "Diabetes mellitus",
    "Type 2 diabetes mellitus without complications": "Diabetes mellitus",
    # --- Limb injury or pain (extremity pain + fractures/abrasions/sprains) ---
    "Pain in unspecified limb":                       "Limb injury or pain",
    "Pain in unspecified knee":                       "Limb injury or pain",
    "Abrasion":                                       "Limb injury or pain",
    "Abrasion of unspecified hand":                   "Limb injury or pain",
    "Disp fx of fifth metatarsal bone":               "Limb injury or pain",
    "Contusion of unspecified hip":                   "Limb injury or pain",
    "Unsp fracture of the lower end of unsp radius":  "Limb injury or pain",
    "Sprain of unspecified ligament of right ankle":  "Limb injury or pain",
    # --- Back / spine pain or strain ---
    "Low back pain":                                  "Back or spine pain",
    "Strain of muscle":                               "Back or spine pain",
    "Sprain of ligaments of lumbar spine":            "Back or spine pain",
    "Contusion of lower back and pelvis":             "Back or spine pain",
    "Unsp fracture of unsp lumbar vertebra":          "Back or spine pain",
    "Sprain of joints and ligaments of oth prt neck": "Back or spine pain",
    # --- Skin / soft-tissue infection ---
    "Cellulitis of unspecified part of limb":             "Skin or soft-tissue infection",
    "Cutaneous abscess of buttock":                       "Skin or soft-tissue infection",
    "Local infection of the skin and subcutaneous tissue":"Skin or soft-tissue infection",
    "Infection following a procedure":                    "Skin or soft-tissue infection",
    # --- UTI / pyelonephritis (lower + upper urinary infection) ---
    "Urinary tract infection":                        "UTI or pyelonephritis",
    "Acute pyelonephritis":                           "UTI or pyelonephritis",
    "Tubulo-interstitial nephritis":                  "UTI or pyelonephritis",
    # --- GI bleed ---
    "Gastrointestinal hemorrhage":                    "GI bleed",
    "Hemorrhage of anus and rectum":                  "GI bleed",
    # --- Allergic reaction ---
    "Allergy":                                        "Allergic reaction",
    "Allergic urticaria":                             "Allergic reaction",
    # --- Head / facial injury ---
    "Contusion of unspecified part of head":          "Head injury",
    "Other specified injuries of head":               "Head injury",
    "Unspecified open wound of other part of head":   "Head injury",
    "Traum subdr hem w/o loss of consciousness":      "Head injury",
    "Fracture of nasal bones":                        "Head injury",
    # --- Moderate (chronic cardiovascular risk; lower-respiratory) ---
    "Essential (primary) hypertension":               "Cardiovascular risk factor",
    "Familial hypercholesterolemia":                  "Cardiovascular risk factor",
    "Athscl heart disease of native coronary artery w/o ang pctrs": "Cardiovascular risk factor",
    "Chronic obstructive pulmonary disease w (acute) exacerbation": "Lower respiratory disease",
    "Pneumonia":                                      "Lower respiratory disease",
}

# Select categories so that AFTER merging there are exactly TOP_N_DISEASES FINAL
# clinical classes. Walk categories in descending patient count; the first
# TOP_N_DISEASES distinct merged labels define the kept clinical classes, then
# keep EVERY category that maps into them (so a low-rank sub-type still joins its
# clinical class instead of being dropped).
def _final_label(c3):
    t = _cat_name.get(c3, "")
    return DISEASE_MERGES.get(t, t) if t else ""

_final_order = []
for _c3 in _cat_patients.index:                 # descending patient count
    _fl = _final_label(_c3)
    if _fl and _fl not in _final_order:
        _final_order.append(_fl)
_kept_finals = set(_final_order[:_TOP_N_DISEASES])
_keep_cats = {c3 for c3 in _cat_patients.index if _final_label(c3) in _kept_finals}
_disease = _disease & _d["cat3"].isin(_keep_cats)
_d["disease_label"] = _d["cat3"].map(_cat_name).fillna("").replace(DISEASE_MERGES)
print(f"Kept {len(_keep_cats)} ICD categories -> {len(_kept_finals)} FINAL clinical "
      f"classes (target TOP_N_DISEASES={_TOP_N_DISEASES}; {len(DISEASE_MERGES)} merge rules applied)")

# ── Comorbidity history (past diagnoses) — LEAK-FREE ──────────────────────────
# For each ED stay, flag disease categories the patient was diagnosed with in
# STRICTLY EARLIER stays (intime < current). This is prior clinical history
# (like home meds), NOT the current target — the current stay's own diagnoses
# are excluded by the temporal window, so there is no leakage. A patient's first
# ever ED visit gets an all-zero history.
_TOP_HX = 30
_hist_rows = _d[_d["cat3"].isin(_cat_name.keys())][["subject_id", "stay_id", "cat3"]]
_stay_cat = (_hist_rows.groupby(["subject_id", "stay_id"])["cat3"]
             .agg(set).reset_index())
_it = edstays[["subject_id", "stay_id", "intime"]].copy()
_it["intime"] = pd.to_datetime(_it["intime"])
_stay_cat = (_stay_cat.merge(_it, on=["subject_id", "stay_id"], how="left")
             .sort_values(["subject_id", "intime"]))
# per-stay set of PRIOR categories (expanding union over earlier stays only)
_prior_sets = {}
for _sid, _grp in _stay_cat.groupby("subject_id", sort=False):
    _acc = set()
    for _stid, _cats in zip(_grp["stay_id"].values, _grp["cat3"].values):
        _prior_sets[_stid] = frozenset(_acc)
        _acc |= _cats
# rank comorbidities by how many stays carry them in prior history, keep top-N
_hx_counter = Counter()
for _s in _prior_sets.values():
    _hx_counter.update(_s)
_keep_hx = [c for c, _ in _hx_counter.most_common(_TOP_HX)]
_hx_colname = {c: "hx_" + re.sub(r'[^a-z0-9]+', '_', _cat_name.get(c, c).lower()).strip('_')
               for c in _keep_hx}
# build per-stay flag columns aligned to df's rows
_stids = df["stay_id"].values
_hx_mat = {col: np.zeros(len(df), dtype=np.int8) for col in dict.fromkeys(_hx_colname.values())}
for _i, _stid in enumerate(_stids):
    _pri = _prior_sets.get(_stid)
    if _pri:
        for _c in _pri:
            _col = _hx_colname.get(_c)
            if _col is not None:
                _hx_mat[_col][_i] = 1
for _col, _arr in _hx_mat.items():
    df[_col] = _arr
print(f"Added {len(_hx_mat)} comorbidity-history columns (hx_*, top-{_TOP_HX}, "
      f"leak-free prior-visit window)")

# --- Symptoms: same standardization as diseases (3-char category + prevalence
# filter + most-frequent canonical name). Collapses "Chest pain"/"Other chest
# pain" -> one R07 category, etc. ---
_MIN_SYMPTOM_PREV = 0   # keep ALL symptom categories (no prevalence filter)
_n_stays = _d["stay_id"].nunique()
_sym_rows = _d[_symptom & _d["icd_title"].ne("") & _d["icd_title"].ne("nan")]
_sym_patients = _sym_rows.drop_duplicates(["stay_id", "cat3"])["cat3"].value_counts()
_keep_sym = set(_sym_patients[_sym_patients >= _MIN_SYMPTOM_PREV * _n_stays].index)
_sym_name = (
    _sym_rows.groupby("cat3")["icd_title"].agg(_category_label).to_dict()
)
_sym_name = {k: v for k, v in _sym_name.items() if v}
_keep_sym = _keep_sym & set(_sym_name.keys())
print(f"Symptom categories kept (>= {_MIN_SYMPTOM_PREV:.4f} prevalence): {len(_keep_sym)}")
_symptom = _symptom & _d["cat3"].isin(_keep_sym)
_d["symptom_label"] = _d["cat3"].map(_sym_name).fillna("")

_d = _d.sort_values(["subject_id", "stay_id", "seq_num"])

def _topn_titles(mask, n, prefix, label_col):
    sub = _d[mask & _d[label_col].ne("") & _d[label_col].ne("nan")]
    # ordered unique labels per stay, then take first n into separate columns
    agg = (sub.groupby(["subject_id", "stay_id"])[label_col]
              .agg(lambda s: list(dict.fromkeys(s)))   # ordered de-dup
              .reset_index())
    for i in range(n):
        agg[f"{prefix}_{i+1}"] = agg[label_col].apply(
            lambda lst: lst[i] if i < len(lst) else "")
    return agg.drop(columns=[label_col])

# both disease and symptom use their 3-char category label
disease_cols = _topn_titles(_disease, 3, "disease", "disease_label")
symptom_cols = _topn_titles(_symptom, 5, "symptom", "symptom_label")

codes_agg = (_d.groupby(["subject_id", "stay_id"])
                .agg(icd_codes=("icd_code", lambda x: "; ".join(x.astype(str))),
                     n_diagnoses=("icd_code", "count"))
                .reset_index())

diag_agg = (codes_agg
            .merge(disease_cols, on=["subject_id", "stay_id"], how="left")
            .merge(symptom_cols, on=["subject_id", "stay_id"], how="left"))
df = df.merge(diag_agg, on=["subject_id", "stay_id"], how="left")

# medrecon: multiple rows per stay → join med names
med_agg = (
    medrecon
    .groupby(["subject_id", "stay_id"])
    .agg(
        medications=("name", lambda x: "; ".join(x.dropna().astype(str))),
        n_medications=("name", "count"),
    )
    .reset_index()
)
df = df.merge(med_agg, on=["subject_id", "stay_id"], how="left")

# pyxis: multiple rows per stay → join dispensed med names
pyxis_agg = (
    pyxis
    .groupby(["subject_id", "stay_id"])
    .agg(
        dispensed_meds=("name", lambda x: "; ".join(x.dropna().astype(str))),
        n_dispensed=("name", "count"),
    )
    .reset_index()
)
df = df.merge(pyxis_agg, on=["subject_id", "stay_id"], how="left")

# convert triage temperature from Fahrenheit to Celsius.
# (vs_* temperature columns were already converted before aggregation above.)
# dataset is 99.8% Fahrenheit; the ~0.1% already in Celsius range are also
# treated as Fahrenheit since they cannot be distinguished reliably at scale
for col in ["temperature"]:
    if col in df.columns:
        df[col] = (df[col] - 32) * 5 / 9

# clamp out-of-range vital signs to NaN (ranges defined in vital_ranges.txt)
VITAL_RANGES = {
    "temperature":   (25.0, 45.0),
    "heartrate":     (5.0,  300.0),
    "resprate":      (1.0,  60.0),
    "o2sat":         (50.0, 100.0),
    "sbp":           (40.0, 300.0),
    "dbp":           (0.0,  200.0),
    "vs_temperature":(25.0, 45.0),
    "vs_heartrate":  (5.0,  300.0),
    "vs_resprate":   (1.0,  60.0),
    "vs_o2sat":      (50.0, 100.0),
    "vs_sbp":        (40.0, 300.0),
    "vs_dbp":        (0.0,  200.0),
}
for col, (lo, hi) in VITAL_RANGES.items():
    if col in df.columns:
        outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        df.loc[(df[col] < lo) | (df[col] > hi), col] = float("nan")
        if outliers:
            print(f"  {col}: {outliers} outlier(s) set to NaN (range [{lo}, {hi}])")

# race grouping (mapping defined in race_grouping.txt)
RACE_MAP = {
    "WHITE":                                    "WHITE",
    "WHITE - RUSSIAN":                          "WHITE",
    "WHITE - BRAZILIAN":                        "WHITE",
    "WHITE - OTHER EUROPEAN":                   "WHITE",
    "WHITE - EASTERN EUROPEAN":                 "WHITE",
    "BLACK/AFRICAN AMERICAN":                   "BLACK",
    "BLACK/AFRICAN":                            "BLACK",
    "BLACK/CAPE VERDEAN":                       "BLACK",
    "BLACK/CARIBBEAN ISLAND":                   "BLACK",
    "HISPANIC OR LATINO":                       "HISPANIC",
    "HISPANIC/LATINO - CUBAN":                  "HISPANIC",
    "HISPANIC/LATINO - PUERTO RICAN":           "HISPANIC",
    "HISPANIC/LATINO - DOMINICAN":              "HISPANIC",
    "HISPANIC/LATINO - GUATEMALAN":             "HISPANIC",
    "HISPANIC/LATINO - SALVADORAN":             "HISPANIC",
    "HISPANIC/LATINO - COLUMBIAN":              "HISPANIC",
    "HISPANIC/LATINO - CENTRAL AMERICAN":       "HISPANIC",
    "HISPANIC/LATINO - MEXICAN":                "HISPANIC",
    "HISPANIC/LATINO - HONDURAN":               "HISPANIC",
    "SOUTH AMERICAN":                           "HISPANIC",
    "PORTUGUESE":                               "HISPANIC",
    "ASIAN":                                    "ASIAN",
    "ASIAN - ASIAN INDIAN":                     "ASIAN",
    "ASIAN - CHINESE":                          "ASIAN",
    "ASIAN - KOREAN":                           "ASIAN",
    "ASIAN - SOUTH EAST ASIAN":                 "ASIAN",
    "AMERICAN INDIAN/ALASKA NATIVE":            "NATIVE",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER":"NATIVE",
    "OTHER":                                    "OTHER",
    "MULTIPLE RACE/ETHNICITY":                  "OTHER",
}
UNKNOWN_RACE = {"UNKNOWN", "PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN"}

before = len(df)
df = df[~df["race"].isin(UNKNOWN_RACE)].copy()
print(f"Race unknown drop: {before} → {len(df)} rows (dropped {before - len(df)})")
df["race"] = df["race"].map(RACE_MAP)

# keep only HOME and ADMITTED dispositions
before = len(df)
df = df[df["disposition"].isin(["HOME", "ADMITTED"])].copy()
print(f"Disposition filter: {before} → {len(df)} rows (dropped {before - len(df)})")

# binary encode disposition: HOME=0, ADMITTED=1
df["disposition"] = df["disposition"].map({"HOME": 0, "ADMITTED": 1})

# drop identifier columns. chiefcomplaint is KEPT here and standardized into
# chiefcomplaint_1..5 multi-hot inputs near the end of the pipeline (it is
# triage free text recorded before diagnosis → a legitimate, non-leaking signal).
# hadm_id: only present for ADMITTED patients, not a feature
# stay_id: visit identifier, not a feature
df = df.drop(columns=["hadm_id", "stay_id"])

# standardize medication names (brand→generic, salt stripping, devices dropped, etc.)
# logic defined in med_standardizer.py
df["medications"] = df["medications"].apply(
    lambda v: clean_med_list(re.sub(r'\s*\[.*?\]', '', str(v))) if pd.notna(v) else "-"
)
df["n_medications"] = df["n_medications"].fillna(0).astype(int)

# pyxis (dispensed_meds) is DROPPED: ED-dispensed drugs are chosen AFTER the
# diagnosis is made (nitrofurantoin -> UTI, albuterol -> asthma), so they leak
# the target. Only home meds (med_*, from medrecon) are kept.
df = df.drop(columns=["dispensed_meds", "n_dispensed", "n_diagnoses"], errors="ignore")
print("Dropped ED-dispensed meds (pyxis) — leakage for diagnosis target")

# acuity: drop rows with missing value, then convert to integer
before = len(df)
df = df.dropna(subset=["acuity"]).copy()
print(f"Acuity filter: {before} → {len(df)} rows (dropped {before - len(df)})")
df["acuity"] = df["acuity"].astype(int)

# LOS: length of stay in hours derived from intime and outtime, then drop raw timestamps
df["intime"] = pd.to_datetime(df["intime"])
df["outtime"] = pd.to_datetime(df["outtime"])
df["los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
df = df.drop(columns=["intime", "outtime"])

# drop rows with missing vital signs
vital_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
              "vs_temperature", "vs_heartrate", "vs_resprate", "vs_o2sat", "vs_sbp", "vs_dbp"]
before = len(df)
df = df.dropna(subset=vital_cols).copy()
print(f"Vital signs filter: {before} → {len(df)} rows (dropped {before - len(df)})")

# arrival_transport: drop UNKNOWN and OTHER rows
before = len(df)
df = df[~df["arrival_transport"].isin(["UNKNOWN", "OTHER"])].copy()
print(f"Arrival transport filter: {before} → {len(df)} rows (dropped {before - len(df)})")

# drop rows with no diagnosis, then remove diagnoses text column (icd_codes retained)
# drop rows with no real disease diagnosis (disease_1 empty) — disease columns
# are the prediction target, so a usable row needs at least one real disease.
before = len(df)
df["disease_1"] = df["disease_1"].fillna("")
df = df[df["disease_1"].astype(str).str.len() > 0].copy()
for i in range(1, 4):
    df[f"disease_{i}"] = df[f"disease_{i}"].fillna("")
for i in range(1, 6):
    df[f"symptom_{i}"] = df[f"symptom_{i}"].fillna("")
print(f"Diagnosis target filter (no real disease): {before} → {len(df)} rows "
      f"(dropped {before - len(df)})")

# SINGLE-DISEASE filter: keep ONLY patients with exactly one real disease
# category (disease_2 empty). This makes the diagnosis target a clean
# single-label classification problem. ~73% of patients with a disease have
# exactly one; the multi-morbid rest (~27%) are dropped.
before = len(df)
df = df[df["disease_2"].astype(str).str.len() == 0].copy()
df = df.drop(columns=["disease_2", "disease_3"])
print(f"Single-disease filter: {before} → {len(df)} rows "
      f"(dropped {before - len(df)} multi-disease patients)")

# ICD-9 → ICD-10 standardization (mapping from icd9_to_icd10_mapping.csv)
icd_mapping = pd.read_csv(os.path.join(RAW_DIR, "icd9_to_icd10_mapping.csv"))
icd_lookup = (
    icd_mapping[icd_mapping["no_map"] == 0]
    .sort_values("approximate")
    .drop_duplicates(subset="icd9_code", keep="first")
    .set_index("icd9_code")["icd10_code"]
    .to_dict()
)

def standardize_icd(val):
    if pd.isna(val):
        return None
    codes = [c.strip() for c in str(val).split(";")]
    result = []
    for c in codes:
        if re.match(r'^[A-Z]', c):
            result.append(c)
        elif c in icd_lookup:
            result.append(icd_lookup[c])
        else:
            return None
    return "; ".join(result)

before = len(df)
df["icd_codes"] = df["icd_codes"].apply(standardize_icd)
df = df.dropna(subset=["icd_codes"]).copy()
print(f"ICD standardization: {before} → {len(df)} rows (dropped {before - len(df)} with unmappable codes)")

# pain: keep only clean numeric values in [0, 10], drop all else
def parse_pain(val):
    if pd.isna(val):
        return None
    try:
        n = float(str(val).strip())
        return n if 0 <= n <= 10 else None
    except:
        return None

before = len(df)
df["pain"] = df["pain"].apply(parse_pain)
df = df.dropna(subset=["pain"]).copy()
print(f"Pain filter: {before} → {len(df)} rows (dropped {before - len(df)})")

# gender, race, arrival_transport: one-hot encoding (after all filters)
df = pd.get_dummies(df, columns=["gender", "race", "arrival_transport"], prefix=["gender", "race", "transport"])

# medications: one-hot encoding for top 300 medications
# patients with no medications ("-") are kept and get all-zero med columns
NO_MED_SENTINEL = "-"

def split_meds(val):
    if pd.isna(val) or str(val).strip() == NO_MED_SENTINEL:
        return []
    return [m.strip() for m in str(val).split(";") if m.strip()]

# count occurrences across all patients
from collections import Counter
med_counter = Counter()
for meds in df["medications"].apply(split_meds):
    med_counter.update(set(meds))  # count per-patient (not per-occurrence)

top300 = [med for med, _ in med_counter.most_common(300)]
top300_set = set(top300)

# drop rows where patient HAS medications but none of them are in top 300
has_meds = df["medications"].apply(lambda v: str(v).strip() != NO_MED_SENTINEL and not pd.isna(v))
has_top300 = df["medications"].apply(lambda v: bool(top300_set & set(split_meds(v))))
before = len(df)
df = df[~has_meds | has_top300].copy()
print(f"Medication filter (no top-300 match): {before} → {len(df)} rows (dropped {before - len(df)})")

# create binary flag columns
for med in top300:
    col = "med_" + re.sub(r'[^a-z0-9]+', '_', med.lower()).strip('_')
    df[col] = df["medications"].apply(lambda v, m=med: int(m in split_meds(v)))

df = df.drop(columns=["medications"])
print(f"Added {len(top300)} medication flag columns (med_*)")

# Therapeutic-class (ATC-like) grouping: dense, clinically-meaningful drug-class
# indicators on top of the sparse med_* one-hots. medclass_<class> = 1 if the
# patient is on ANY drug of that class (antidiabetic, anticonvulsant, ...).
from med_classes import med_class as _med_class
_med_cols = [c for c in df.columns if c.startswith("med_")]
_class_of = {c: _med_class(c[len("med_"):]) for c in _med_cols}
_classes = sorted({v for v in _class_of.values() if v})
for _cls in _classes:
    _members = [c for c, v in _class_of.items() if v == _cls]
    df["medclass_" + _cls] = (df[_members].sum(axis=1) > 0).astype(int)
print(f"Added {len(_classes)} therapeutic-class columns (medclass_*): {_classes}")

# (pyxis already dropped above — ED-dispensed meds leak the diagnosis target.)

# ── Chief complaint (triage free text) → standardized multi-hot inputs ─────────
# Recorded at TRIAGE, before any diagnosis → legitimate, non-leaking predictor.
# Standardize via chiefcomplaint_standardizer (abbreviation expansion + laterality
# stripping), keep canonical complaints with >= 0.2% prevalence in the final
# dataset, and emit ordered chiefcomplaint_1..5 columns (parallel to symptom_*).
_CC_MIN_PREV = 0.002
_cc_lists = df["chiefcomplaint"].apply(clean_complaint_list)
_cc_counter = Counter()
for _lst in _cc_lists:
    _cc_counter.update(set(_lst))
_cc_n = len(df)
_keep_cc = {t for t, c in _cc_counter.items() if c >= _CC_MIN_PREV * _cc_n}
print(f"Chief-complaint tokens kept (>= {_CC_MIN_PREV:.3f} prevalence): "
      f"{len(_keep_cc)} of {len(_cc_counter)} canonical tokens")
_CC_MAX = 5
_cc_filtered = _cc_lists.apply(lambda lst: [t for t in lst if t in _keep_cc])
for _i in range(_CC_MAX):
    df[f"chiefcomplaint_{_i+1}"] = _cc_filtered.apply(
        lambda lst, i=_i: lst[i] if i < len(lst) else "")
df = df.drop(columns=["chiefcomplaint"])

# Round all floating-point columns to 3 decimals (vitals/LOS come out with long
# repeating decimals from the F->C conversion and averaging).
_float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns
df[_float_cols] = df[_float_cols].round(3)
print(f"Rounded {len(_float_cols)} float columns to 3 decimals")

def save_dataset(df, name):
    # NO sampling / balancing. The training set is ONE visit per DISTINCT patient
    # (drop_duplicates on subject_id) — this prevents subject-level leakage while
    # keeping the WHOLE cohort. Class imbalance is handled at model time via class
    # weights, not by down-sampling. An all-visits file is also written for any
    # multi-visit analysis.
    distinct = (
        df.sort_values("subject_id")
        .drop_duplicates(subset="subject_id", keep="first")
        .reset_index(drop=True)
    )
    all_path = os.path.join(BASE_DIR, f"{name}_all_visits.csv")
    df.to_csv(all_path, index=False)
    print(f"Saved: {all_path}  shape={df.shape}  (all visits, reference)")

    out_path = os.path.join(BASE_DIR, f"{name}.csv")
    distinct.to_csv(out_path, index=False)
    vc = distinct["disease_1"].value_counts()
    print(f"Saved: {out_path}  shape={distinct.shape}  "
          f"distinct_patients={distinct['subject_id'].nunique()}  "
          f"classes={len(vc)}  (max class={vc.max()}, min class={vc.min()})  "
          f"— NO sampling, full distinct-patient cohort")

# Drop columns that must never reach the model:
#   icd_codes   - the diagnosis target was derived from it (direct leakage)
#   los_hours   - length of stay, known only AFTER the visit ends (post-outcome)
#   disposition - the old HOME/ADMITTED target, decided after the diagnosis
# subject_id is KEPT: the GNN reads it to build a SUBJECT-AWARE split (no patient
# appears in both train and test); it is excluded from features downstream.
_drop_cols = ["icd_codes", "los_hours", "disposition"]
df_clean = df.drop(columns=[c for c in _drop_cols if c in df.columns])
print(f"Dropped from saved CSV: {[c for c in _drop_cols if c in df.columns]}")
_out_name = "merged_ed" if _TOP_N_DISEASES == 30 else f"merged_ed_top{_TOP_N_DISEASES}"
save_dataset(df_clean, _out_name)
