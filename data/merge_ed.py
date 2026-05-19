import pandas as pd
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from med_standardizer import clean_med_list

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# vitalsign: multiple rows per stay → aggregate numerics
vs_agg = (
    vitalsign
    .groupby(["subject_id", "stay_id"])
    .agg(
        vs_temperature=("temperature", "mean"),
        vs_heartrate=("heartrate", "mean"),
        vs_resprate=("resprate", "mean"),
        vs_o2sat=("o2sat", "mean"),
        vs_sbp=("sbp", "mean"),
        vs_dbp=("dbp", "mean"),
        vs_count=("charttime", "count"),
    )
    .reset_index()
)
df = df.merge(vs_agg, on=["subject_id", "stay_id"], how="left")

# diagnosis: multiple rows per stay → join icd titles as semicolon-separated string
diag_agg = (
    diagnosis
    .groupby(["subject_id", "stay_id"])
    .agg(
        diagnoses=("icd_title", lambda x: "; ".join(x.dropna().astype(str))),
        icd_codes=("icd_code", lambda x: "; ".join(x.dropna().astype(str))),
        n_diagnoses=("icd_code", "count"),
    )
    .reset_index()
)
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

# convert temperature from Fahrenheit to Celsius
# dataset is 99.8% Fahrenheit; the ~0.1% already in Celsius range are also
# treated as Fahrenheit since they cannot be distinguished reliably at scale
for col in ["temperature", "vs_temperature"]:
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

# drop identifier and uninformative columns
# hadm_id: only present for ADMITTED patients, not a feature
# chiefcomplaint: 52k unique free-text values, redundant with icd_codes
# stay_id: visit identifier, not a feature
df = df.drop(columns=["hadm_id", "chiefcomplaint", "stay_id"])

# standardize medication names (brand→generic, salt stripping, devices dropped, etc.)
# logic defined in med_standardizer.py
df["medications"] = df["medications"].apply(
    lambda v: clean_med_list(re.sub(r'\s*\[.*?\]', '', str(v))) if pd.notna(v) else "-"
)
df["n_medications"] = df["n_medications"].fillna(0).astype(int)

# drop dispensed_meds — redundant with medications column
# drop n_medications and n_diagnoses — count columns add no signal beyond the feature columns
df = df.drop(columns=["dispensed_meds", "n_dispensed", "n_medications", "n_diagnoses"])

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
before = len(df)
df = df.dropna(subset=["diagnoses"]).copy()
print(f"Diagnoses filter: {before} → {len(df)} rows (dropped {before - len(df)})")
df = df.drop(columns=["diagnoses"])

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

def save_with_sample(df, name):
    out_path = os.path.join(BASE_DIR, f"merged_ed_{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  shape={df.shape}")

    SAMPLE_SIZE = 20_000
    sample = (
        df.groupby("disposition", group_keys=False)
        .apply(lambda x: x.sample(frac=SAMPLE_SIZE / len(df), random_state=42))
        .reset_index(drop=True)
    )
    sample_path = os.path.join(BASE_DIR, f"merged_ed_{name}_sample_20k.csv")
    sample.to_csv(sample_path, index=False)
    print(f"Saved: {sample_path}  shape={sample.shape}  disposition={sample['disposition'].value_counts().to_dict()}")

# with ICD codes
save_with_sample(df, "with_icd")

# without ICD codes
df_no_icd = df.drop(columns=["icd_codes"])
save_with_sample(df_no_icd, "no_icd")
