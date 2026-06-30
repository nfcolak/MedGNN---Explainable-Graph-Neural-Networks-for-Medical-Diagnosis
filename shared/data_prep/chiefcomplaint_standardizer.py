"""
Chief-complaint standardizer for MIMIC-IV ED triage free text.

Parallel to med_standardizer.py. The raw `chiefcomplaint` field is messy free
text: comma-separated multi-complaints, ALL-CAPS / mixed case, heavy medical
shorthand (SOB, ETOH, SI, N/V, MVC, BRBPR, s/p Fall ...) and left/right
laterality that does not change the underlying disease.

Pipeline (mirrors how symptoms are standardized: normalize -> canonicalize ->
the caller keeps the top-N and multi-hot encodes):

    1. split on comma          "Abd pain, N/V"  -> ["abd pain", "n/v"]
    2. normalize text          lowercase, collapse spaces, strip punctuation
    3. strip laterality        "r flank pain"/"l flank pain" -> "flank pain"
    4. canonicalize / expand   abbreviation + synonym dictionary, one token may
                               expand to several ("n/v" -> nausea, vomiting)
    5. drop junk               "___", "unknown-cc", "" -> nothing

Public API:
    standardize_complaint(token)  -> list[str]   (0, 1, or many canonical tokens)
    clean_complaint_list(raw)     -> list[str]   (ordered, de-duplicated)

NOTE: chiefcomplaint is recorded at triage (the door), BEFORE any diagnosis, so
it is a legitimate, non-leaking predictor of the disease target.
"""

import re

# ── junk / non-informative tokens → drop entirely ─────────────────────────────
DROP_TOKENS = {
    "", "___", "unknown", "unknown-cc", "unk", "n/a", "na", "none",
    "see hpi", "see note", "other", "?", "eval", "evaluation",
}

# ── laterality / region qualifiers stripped from the FRONT of a token ──────────
# "r flank pain" -> "flank pain", "left knee pain" -> "knee pain",
# "rlq abdominal pain" -> "abdominal pain". Disease label rarely depends on side.
# NOTE: longer / multi-word alternatives MUST come first, otherwise "right"
# matches before "right sided" and leaves a dangling "sided ...".
_LATERALITY = (
    r"^(?:right\s+sided|left\s+sided|bilateral|bilat|sided|"
    r"rlq|ruq|llq|luq|rt|lt|bil|right|left|r|l|b|rl)\s+"
)
_LAT_RE = re.compile(_LATERALITY)

# region words to strip when they merely localise a generic pain
_REGION_PREFIX_RE = re.compile(r"^(?:lower|upper|mid|central|generalized)\s+"
                               r"(?=(?:abdominal pain|back pain|abd pain|"
                               r"extremity pain|chest pain))")

# ── abbreviation / synonym → canonical token(s) ───────────────────────────────
# value may be a string (1:1) or a list (1:many, e.g. n/v -> nausea + vomiting)
ABBREV = {
    # --- abdominal ---
    "abd pain": "abdominal pain", "abd": "abdominal pain",
    "abdominal pain": "abdominal pain",
    "epigastric pain": "abdominal pain",
    "luq abd pain": "abdominal pain",
    "abdominal distention": "abdominal distention",
    # --- chest / cardiac ---
    "cp": "chest pain", "chest pain": "chest pain",
    # parentheses are stripped by _normalize, so key the normalized form too
    "chest pain (cardiac features)": "chest pain",
    "chest pain cardiac features": "chest pain",
    "palpitations": "palpitations",
    "nstemi": "acute coronary syndrome", "stemi": "acute coronary syndrome",
    "acs": "acute coronary syndrome",
    "atrial fibrillation": "atrial fibrillation", "afib": "atrial fibrillation",
    "tachycardia": "tachycardia", "bradycardia": "bradycardia",
    "cardiac arrest": "cardiac arrest",
    # --- respiratory ---
    "sob": "dyspnea", "short of breath": "dyspnea",
    "shortness of breath": "dyspnea", "dyspnea": "dyspnea",
    "doe": "dyspnea", "dyspnea on exertion": "dyspnea",
    "respiratory distress": "dyspnea",
    "cough": "cough", "productive cough": "cough",
    "hemoptysis": "hemoptysis", "hypoxia": "hypoxia",
    "asthma exacerbation": "asthma exacerbation",
    "pneumonia": "pneumonia", "ili": "influenza-like illness",
    "sore throat": "sore throat", "difficulty swallowing": "dysphagia",
    "dysphagia": "dysphagia",
    # --- neuro ---
    "ha": "headache", "headache": "headache",
    "altered mental status": "altered mental status",
    "ams": "altered mental status", "ms": "altered mental status",
    "confusion": "altered mental status", "lethargy": "altered mental status",
    "unresponsive": "altered mental status", "found down": "altered mental status",
    "syncope": "syncope", "presyncope": "presyncope",
    "lightheaded": "dizziness", "dizziness": "dizziness",
    "seizure": "seizure", "slurred speech": "focal neuro deficit",
    "numbness": "focal neuro deficit", "weakness": "weakness",
    "arm numbness": "focal neuro deficit", "leg numbness": "focal neuro deficit",
    "facial numbness": "focal neuro deficit", "facial droop": "focal neuro deficit",
    "tingling": "focal neuro deficit",
    "focal weakness": "focal neuro deficit", "leg weakness": "weakness",
    "unsteady gait": "gait disturbance", "unable to ambulate": "gait disturbance",
    "visual changes": "visual changes",
    "cva": "stroke", "tia": "stroke", "stroke": "stroke",
    "sah": "subarachnoid hemorrhage", "sdh": "subdural hemorrhage",
    "ich": "intracranial hemorrhage",
    # --- psych / substance ---
    "si": "suicidal ideation", "suicidal ideation": "suicidal ideation",
    "suicidal ideations": "suicidal ideation", "sih": "suicidal ideation",
    "hi": "homicidal ideation",
    "etoh": "alcohol intoxication", "intoxication": "alcohol intoxication",
    "detox": "alcohol intoxication", "withdrawal": "alcohol withdrawal",
    "overdose": "overdose", "od": "overdose", "ingestion": "overdose",
    "substance use": "substance use", "substance misuse/intoxication": "substance use",
    "substance abuse": "substance use",
    "anxiety": "anxiety", "depression": "depression",
    "agitation": "agitation", "hallucinations": "psychosis",
    "psych": "psychiatric evaluation", "psych eval": "psychiatric evaluation",
    "hallucinating": "psychosis",
    # --- GI ---
    "n/v": ["nausea", "vomiting"], "nausea": "nausea", "vomiting": "vomiting",
    "n/v/d": ["nausea", "vomiting", "diarrhea"],
    "diarrhea": "diarrhea", "constipation": "constipation",
    "brbpr": "rectal bleeding", "rectal bleeding": "rectal bleeding",
    "melena": "gi bleed", "hematemesis": "gi bleed", "gi bleed": "gi bleed",
    "sbo": "bowel obstruction", "bowel obstruction": "bowel obstruction",
    "jaundice": "jaundice", "rectal pain": "rectal pain",
    # --- GU ---
    "dysuria": "dysuria", "hematuria": "hematuria",
    "urinary retention": "urinary retention",
    "urinary frequency": "urinary frequency", "uti": "dysuria",
    "flank pain": "flank pain", "testicular pain": "testicular pain",
    "vaginal bleeding": "vaginal bleeding", "pelvic pain": "pelvic pain",
    "pregnant": "pregnancy related", "pregnancy": "pregnancy related",
    # --- trauma / MSK ---
    "s/p fall": "fall", "fall": "fall", "found on floor": "fall",
    "mvc": "motor vehicle collision", "s/p mvc": "motor vehicle collision",
    "ped struck": "pedestrian struck", "assault": "assault",
    "head injury": "head injury", "head trauma": "head injury",
    "head lac": "laceration", "laceration": "laceration",
    "back pain": "back pain", "lower back pain": "back pain",
    "neck pain": "neck pain",
    "leg pain": "leg pain", "knee pain": "knee pain", "foot pain": "foot pain",
    "arm pain": "arm pain", "shoulder pain": "shoulder pain",
    "hip pain": "hip pain", "ankle pain": "ankle pain",
    "wrist pain": "wrist pain", "hand pain": "hand pain",
    "ankle injury": "extremity injury", "hand injury": "extremity injury",
    "finger injury": "extremity injury", "finger laceration": "laceration",
    "leg swelling": "extremity swelling", "leg pain/swelling": "extremity swelling",
    "extremity pain": "extremity pain", "body pain": "generalized pain",
    "body aches": "generalized pain", "pain": "generalized pain",
    "dog bite": "animal bite", "bite": "animal bite",
    "dental pain": "dental pain", "jaw pain": "jaw pain",
    "ear pain": "ear pain", "eye pain": "eye pain", "facial swelling": "facial swelling",
    "epistaxis": "epistaxis", "rib pain": "rib pain",
    # --- skin / infection ---
    "rash": "rash", "allergic reaction": "allergic reaction",
    "cellulitis": "cellulitis", "abscess": "abscess",
    "wound eval": "wound evaluation", "wound check": "wound evaluation",
    "chills": "fever", "fever": "fever",
    # --- metabolic / heme ---
    "hyperglycemia": "hyperglycemia", "hypoglycemia": "hypoglycemia",
    "hypertension": "hypertension", "htn": "hypertension",
    "hypotension": "hypotension", "hyperkalemia": "hyperkalemia",
    "anemia": "anemia", "weakness/fatigue": "fatigue", "fatigue": "fatigue",
    "failure to thrive": "failure to thrive", "ftt": "failure to thrive",
    # --- workup / administrative (kept; weakly informative) ---
    "abnormal labs": "abnormal labs", "abnormal ct": "abnormal imaging",
    "abnormal mri": "abnormal imaging", "abnormal ekg": "abnormal ekg",
    "transfer": "transfer", "med refill": "medication refill",
    "suture removal": "suture removal", "gtube eval": "device evaluation",
    "rci": "transfer",
}


def _normalize(token: str) -> str:
    t = str(token).strip().lower()
    t = t.replace("&", " and ")
    # keep slashes for n/v, s/p; drop other punctuation
    t = re.sub(r"[^a-z0-9/+\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def standardize_complaint(token: str):
    """Return a list of canonical complaint tokens for ONE raw token
    (0 tokens if junk, usually 1, sometimes several)."""
    t = _normalize(token)
    if t in DROP_TOKENS:
        return []

    # exact dictionary hit (before stripping) — catches "s/p fall", "n/v" ...
    if t in ABBREV:
        v = ABBREV[t]
        return list(v) if isinstance(v, list) else [v]

    # strip laterality / region qualifiers, retry the dictionary
    stripped = _LAT_RE.sub("", t)
    stripped = _REGION_PREFIX_RE.sub("", stripped).strip()
    if stripped != t:
        if stripped in DROP_TOKENS:
            return []
        if stripped in ABBREV:
            v = ABBREV[stripped]
            return list(v) if isinstance(v, list) else [v]
        t = stripped

    # no dictionary entry: keep the normalized (laterality-stripped) token as-is.
    # The caller's top-N frequency filter drops the long noisy tail.
    return [t] if t and t not in DROP_TOKENS else []


def clean_complaint_list(raw: str):
    """Split a raw chiefcomplaint string on commas, standardize each part, and
    return an ordered, de-duplicated list of canonical complaint tokens."""
    if raw is None:
        return []
    s = str(raw)
    if not s.strip() or s.strip().lower() in DROP_TOKENS:
        return []
    out = []
    for part in s.split(","):
        for tok in standardize_complaint(part):
            if tok not in out:
                out.append(tok)
    return out


# ── standalone inspection: run BEFORE wiring into merge_ed.py ──────────────────
if __name__ == "__main__":
    import os
    import pandas as pd

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")  # <root>/data
    path = os.path.join(data_dir, "Original CSVs", "triage.csv")
    cc = pd.read_csv(path, usecols=["chiefcomplaint"], low_memory=False)
    raw = cc["chiefcomplaint"].dropna().astype(str)

    # 1) how the top raw tokens map
    toks = raw.str.split(",").explode().str.strip()
    top_raw = toks.str.lower().value_counts().head(60)
    print("=== top-60 raw tokens -> canonical ===")
    for name, c in top_raw.items():
        mapped = standardize_complaint(name)
        print(f"{c:>7}  {name:<28} -> {mapped}")

    # 2) canonical vocabulary distribution
    from collections import Counter
    counter = Counter()
    for v in raw:
        counter.update(set(clean_complaint_list(v)))
    print(f"\n=== canonical vocabulary: {len(counter)} distinct tokens ===")
    print("--- top 100 canonical complaints (token : patient_count) ---")
    for name, c in counter.most_common(100):
        print(f"{c:>7}  {name}")

    # 3) how many patients have at least one canonical complaint
    nonempty = sum(1 for v in raw if clean_complaint_list(v))
    print(f"\npatients with >=1 canonical complaint: {nonempty}/{len(raw)} "
          f"({nonempty/len(raw):.1%})")
