"""
Therapeutic-class (ATC-like) grouping for home medications.

Collapses the ~300 sparse one-hot `med_<generic>` columns into a small set of
dense, clinically-meaningful drug-class indicators (`medclass_<class>`). A drug
class is a much stronger, less sparse signal than an individual drug AND maps
cleanly to disease groups (antidiabetic -> diabetes, anticonvulsant -> epilepsy,
bronchodilator -> asthma/COPD, ...), so it both densifies the matrix and stays
interpretable.

MED_CLASS maps a standardized generic name (as produced by med_standardizer,
i.e. the token after the `med_` prefix with non-alphanumerics -> '_') to one
therapeutic class. Drugs not listed simply get no class (kept only as med_*).
"""

# generic (med_<generic>) -> therapeutic class
MED_CLASS = {
    # --- antidiabetic ---
    "metformin": "antidiabetic", "insulin_glargine": "antidiabetic",
    "insulin_lispro": "antidiabetic", "insulin_aspart": "antidiabetic",
    "insulin_detemir": "antidiabetic", "insulin_regular_human": "antidiabetic",
    "glipizide": "antidiabetic", "glyburide": "antidiabetic",
    "glimepiride": "antidiabetic", "sitagliptin": "antidiabetic",
    "blood_sugar_diagnostic": "antidiabetic", "pioglitazone": "antidiabetic",
    # --- antihypertensive / cardiac ---
    "lisinopril": "antihypertensive", "enalapril": "antihypertensive",
    "amlodipine": "antihypertensive", "nifedipine": "antihypertensive",
    "metoprolol": "antihypertensive", "atenolol": "antihypertensive",
    "carvedilol": "antihypertensive", "propranolol": "antihypertensive",
    "nadolol": "antihypertensive", "labetalol": "antihypertensive",
    "losartan": "antihypertensive", "valsartan": "antihypertensive",
    "irbesartan": "antihypertensive", "olmesartan": "antihypertensive",
    "diltiazem": "antihypertensive", "verapamil": "antihypertensive",
    "hydralazine": "antihypertensive", "clonidine": "antihypertensive",
    "doxazosin": "antihypertensive", "terazosin": "antihypertensive",
    "prazosin": "antihypertensive", "lisinopril_hydrochlorothiazide": "antihypertensive",
    "losartan_hydrochlorothiazide": "antihypertensive", "isosorbide_mononitrate": "antihypertensive",
    "nitroglycerin": "antihypertensive", "amiodarone": "antihypertensive",
    "digoxin": "antihypertensive", "midodrine": "antihypertensive",
    # --- diuretic ---
    "furosemide": "diuretic", "hydrochlorothiazide": "diuretic",
    "spironolactone": "diuretic", "torsemide": "diuretic",
    "chlorthalidone": "diuretic", "metolazone": "diuretic",
    "triamterene_hydrochlorothiazid": "diuretic",
    # --- lipid-lowering ---
    "atorvastatin": "lipid_lowering", "simvastatin": "lipid_lowering",
    "rosuvastatin": "lipid_lowering", "pravastatin": "lipid_lowering",
    "lovastatin": "lipid_lowering", "ezetimibe": "lipid_lowering",
    "fenofibrate": "lipid_lowering",
    # --- anticoagulant / antiplatelet ---
    "warfarin": "anticoagulant", "apixaban": "anticoagulant",
    "rivaroxaban": "anticoagulant", "enoxaparin": "anticoagulant",
    "clopidogrel": "antiplatelet", "aspirin": "antiplatelet",
    "bayer_aspirin": "antiplatelet",
    # --- opioid ---
    "oxycodone": "opioid", "hydromorphone": "opioid", "morphine": "opioid",
    "tramadol": "opioid", "fentanyl": "opioid", "methadone": "opioid",
    "hydrocodone_acetaminophen": "opioid", "oxycodone_acetaminophen": "opioid",
    "acetaminophen_codeine": "opioid", "oxycontin": "opioid", "percocet": "opioid",
    "suboxone": "opioid", "buprenorphine": "opioid",
    # --- antidepressant ---
    "sertraline": "antidepressant", "citalopram": "antidepressant",
    "escitalopram": "antidepressant", "fluoxetine": "antidepressant",
    "paroxetine": "antidepressant", "venlafaxine": "antidepressant",
    "duloxetine": "antidepressant", "bupropion": "antidepressant",
    "mirtazapine": "antidepressant", "trazodone": "antidepressant",
    "amitriptyline": "antidepressant", "nortriptyline": "antidepressant",
    "doxepin": "antidepressant",
    # --- antipsychotic / mood ---
    "quetiapine": "antipsychotic", "risperidone": "antipsychotic",
    "olanzapine": "antipsychotic", "aripiprazole": "antipsychotic",
    "haloperidol": "antipsychotic", "lithium_carbonate": "antipsychotic",
    # --- benzodiazepine / anxiolytic ---
    "lorazepam": "benzodiazepine", "alprazolam": "benzodiazepine",
    "clonazepam": "benzodiazepine", "diazepam": "benzodiazepine",
    "buspirone": "benzodiazepine",
    # --- anticonvulsant ---
    "levetiracetam": "anticonvulsant", "gabapentin": "anticonvulsant",
    "pregabalin": "anticonvulsant", "lamotrigine": "anticonvulsant",
    "topiramate": "anticonvulsant", "divalproex": "anticonvulsant",
    "valproic_acid": "anticonvulsant", "carbamazepine": "anticonvulsant",
    "oxcarbazepine": "anticonvulsant", "phenytoin_extended": "anticonvulsant",
    "lacosamide": "anticonvulsant", "zonisamide": "anticonvulsant",
    # --- bronchodilator / respiratory ---
    "albuterol": "respiratory", "fluticasone": "respiratory",
    "tiotropium": "respiratory", "budesonide_formoterol": "respiratory",
    "montelukast": "respiratory", "ipratropium": "respiratory",
    "fluticasone_salmeterol": "respiratory", "advair_diskus": "respiratory",
    "ipratropium_albuterol": "respiratory", "spiriva_with_handihaler": "respiratory",
    "budesonide": "respiratory", "tiotropium_bromide": "respiratory",
    # --- acid suppression ---
    "omeprazole": "acid_suppression", "pantoprazole": "acid_suppression",
    "esomeprazole": "acid_suppression", "lansoprazole": "acid_suppression",
    "ranitidine": "acid_suppression", "famotidine": "acid_suppression",
    "esomeprazole_magnesium": "acid_suppression", "prilosec_otc": "acid_suppression",
    # --- corticosteroid (systemic) ---
    "prednisone": "corticosteroid", "prednisolone": "corticosteroid",
    "dexamethasone": "corticosteroid", "hydrocortisone": "corticosteroid",
    # --- antibiotic ---
    "amoxicillin": "antibiotic", "azithromycin": "antibiotic",
    "ciprofloxacin": "antibiotic", "cephalexin": "antibiotic",
    "doxycycline_hyclate": "antibiotic", "sulfamethoxazole_trimethoprim": "antibiotic",
    "levofloxacin": "antibiotic", "clindamycin": "antibiotic",
    "metronidazole": "antibiotic", "amoxicillin_pot_clavulanate": "antibiotic",
    "vancomycin": "antibiotic", "acyclovir": "antibiotic", "valacyclovir": "antibiotic",
    # --- thyroid ---
    "levothyroxine": "thyroid",
    # --- stimulant / ADHD ---
    "amphetamine_salts": "stimulant", "methylphenidate": "stimulant",
    "dextroamphetamine_amphetamine": "stimulant", "lisdexamfetamine": "stimulant",
}


def med_class(generic: str):
    """Return the therapeutic class for a standardized generic, or None."""
    return MED_CLASS.get(generic)
