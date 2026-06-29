import re

# ── 1. Medical devices & supplies → drop entirely ─────────────────────────────
DEVICE_KEYWORDS = {
    "lancet", "syringe", "needle", "pen needle", "blood-glucose meter",
    "glucose meter", "freestyle lite strips", "freestyle lancets",
    "insulin needles", "insulin syringe", "infusion set", "infusion pump",
    "nebulizer", "nebulizer accessories", "nebulizer & compressor",
    "spacer", "inhaler spacer", "iv prep wipes", "iv equipment",
    "catheter", "wheelchair", "crutch", "walker", "brace", "splint",
    "bandage", "foam bandage", "dressing", "wound care", "heating pad",
    "diaper", "brief,adult", "medical supply", "miscellaneous medical",
    "water for injection", "sterile water for injection",
    "bd lo-dose", "minimed infusion", "relion ultra thin",
    "pen needles", "pen needle", "alcohol prep", "alcohol swabs",
    "test strip", "glucose test", "contour", "accu-chek",
    "one touch ultra", "one touch", "truemetrix",
}

# ── 2. Generic/unclear entries → drop ─────────────────────────────────────────
GENERIC_DESCRIPTORS = {
    "food supplement, lactose-free", "food supply, lactose-free",
    "nutritional supplements", "nutritional supplement",
    "nut.tx.gluc.intol,lac-free,soy", "once daily",
    "non-aspirin pain relief", "8 hour pain reliever",
    "pain relief", "misc. supplement",
}

# ── 3. Brand → Generic ────────────────────────────────────────────────────────
BRAND_TO_GENERIC = {
    # analgesics
    "tylenol": "acetaminophen", "mapap": "acetaminophen",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nuprin": "ibuprofen",
    "aleve": "naproxen", "naprosyn": "naproxen", "anaprox": "naproxen",
    "aspir-81": "aspirin", "ecotrin": "aspirin", "bufferin": "aspirin",
    "excedrin": "acetaminophen", "dolophine": "methadone",
    "ultram": "tramadol", "nucynta": "tapentadol",
    "dilaudid": "hydromorphone", "demerol": "meperidine",
    "duragesic": "fentanyl",
    # antihypertensives
    "zestril": "lisinopril", "prinivil": "lisinopril",
    "altace": "ramipril", "vasotec": "enalapril",
    "mavik": "trandolapril", "accupril": "quinapril",
    "lotensin": "benazepril", "monopril": "fosinopril",
    "norvasc": "amlodipine", "cardizem": "diltiazem",
    "tiazac": "diltiazem", "cartia": "diltiazem",
    "calan": "verapamil", "isoptin": "verapamil",
    "toprol xl": "metoprolol", "toprol": "metoprolol",
    "lopressor": "metoprolol", "tenormin": "atenolol",
    "coreg": "carvedilol", "bystolic": "nebivolol",
    "inderal": "propranolol", "innopran": "propranolol",
    "zebeta": "bisoprolol", "sectral": "acebutolol",
    "lasix": "furosemide", "edecrin": "ethacrynic acid",
    "demadex": "torsemide", "bumex": "bumetanide",
    "aldactone": "spironolactone", "inspra": "eplerenone",
    "microzide": "hydrochlorothiazide", "hydrodiuril": "hydrochlorothiazide",
    "cozaar": "losartan", "hyzaar": "losartan-hydrochlorothiazide",
    "diovan": "valsartan", "atacand": "candesartan",
    "benicar": "olmesartan", "avapro": "irbesartan",
    "micardis": "telmisartan", "teveten": "eprosartan",
    # statins
    "lipitor": "atorvastatin", "crestor": "rosuvastatin",
    "zocor": "simvastatin", "pravachol": "pravastatin",
    "mevacor": "lovastatin", "lescol": "fluvastatin",
    "livalo": "pitavastatin",
    # anticoagulants
    "coumadin": "warfarin", "jantoven": "warfarin",
    "eliquis": "apixaban", "xarelto": "rivaroxaban",
    "pradaxa": "dabigatran", "savaysa": "edoxaban",
    "lovenox": "enoxaparin", "fragmin": "dalteparin",
    "arixtra": "fondaparinux",
    # antidiabetics
    "glucophage": "metformin", "glumetza": "metformin",
    "actos": "pioglitazone", "avandia": "rosiglitazone",
    "januvia": "sitagliptin", "onglyza": "saxagliptin",
    "tradjenta": "linagliptin", "nesina": "alogliptin",
    "invokana": "canagliflozin", "farxiga": "dapagliflozin",
    "jardiance": "empagliflozin", "victoza": "liraglutide",
    "ozempic": "semaglutide", "byetta": "exenatide",
    "trulicity": "dulaglutide",
    # insulins
    "lantus": "insulin glargine", "basaglar": "insulin glargine",
    "toujeo": "insulin glargine", "levemir": "insulin detemir",
    "tresiba": "insulin degludec", "humalog": "insulin lispro",
    "admelog": "insulin lispro", "novolog": "insulin aspart",
    "fiasp": "insulin aspart", "apidra": "insulin glulisine",
    "humulin n": "nph insulin", "novolin n": "nph insulin",
    "humulin r": "insulin regular human", "novolin r": "insulin regular human",
    "humulin 70/30": "nph-regular insulin",
    # GI drugs
    "prilosec": "omeprazole", "zegerid": "omeprazole",
    "nexium": "esomeprazole", "dexilant": "dexlansoprazole",
    "prevacid": "lansoprazole", "protonix": "pantoprazole",
    "aciphex": "rabeprazole", "kapidex": "dexlansoprazole",
    "zantac": "ranitidine", "pepcid": "famotidine",
    "tagamet": "cimetidine", "axid": "nizatidine",
    "carafate": "sucralfate", "reglan": "metoclopramide",
    "colace": "docusate", "miralax": "polyethylene glycol 3350",
    "dulcolax": "bisacodyl", "correctol": "bisacodyl",
    "senokot": "senna", "ex-lax": "senna",
    "imodium": "loperamide", "kaopectate": "bismuth subsalicylate",
    "pepto-bismol": "bismuth subsalicylate",
    "maalox": "aluminum hydroxide-magnesium hydroxide",
    "mylanta": "aluminum hydroxide-magnesium hydroxide",
    "tums": "calcium carbonate",
    # antibiotics
    "zithromax": "azithromycin", "zpack": "azithromycin",
    "augmentin": "amoxicillin-clavulanate",
    "bactrim": "sulfamethoxazole-trimethoprim",
    "cipro": "ciprofloxacin", "levaquin": "levofloxacin",
    "avelox": "moxifloxacin", "floxin": "ofloxacin",
    "keflex": "cephalexin", "omnicef": "cefdinir",
    "rocephin": "ceftriaxone", "suprax": "cefixime",
    "flagyl": "metronidazole", "cleocin": "clindamycin",
    "zyvox": "linezolid", "cubicin": "daptomycin",
    "vibramycin": "doxycycline", "doryx": "doxycycline",
    "minocin": "minocycline", "sumycin": "tetracycline",
    "diflucan": "fluconazole", "sporanox": "itraconazole",
    "vfend": "voriconazole", "noxafil": "posaconazole",
    "tamiflu": "oseltamivir", "valtrex": "valacyclovir",
    "zovirax": "acyclovir", "famvir": "famciclovir",
    # respiratory
    "proair hfa": "albuterol", "ventolin hfa": "albuterol",
    "proventil hfa": "albuterol", "proventil": "albuterol",
    "ventolin": "albuterol", "proair": "albuterol",
    "flovent hfa": "fluticasone", "flovent": "fluticasone",
    "flonase": "fluticasone", "veramyst": "fluticasone furoate",
    "rhinocort": "budesonide", "pulmicort": "budesonide",
    "qvar": "beclomethasone", "alvesco": "ciclesonide",
    "atrovent hfa": "ipratropium", "atrovent": "ipratropium",
    "spiriva": "tiotropium", "incruse": "umeclidinium",
    "advair hfa": "fluticasone-salmeterol", "advair": "fluticasone-salmeterol",
    "symbicort": "budesonide-formoterol",
    "dulera": "mometasone-formoterol",
    "breo": "fluticasone furoate-vilanterol",
    "combivent": "ipratropium-albuterol",
    "striverdi": "olodaterol", "serevent": "salmeterol",
    "foradil": "formoterol", "brovana": "arformoterol",
    "singulair": "montelukast", "accolate": "zafirlukast",
    "tessalon": "benzonatate", "robitussin": "guaifenesin",
    "mucinex": "guaifenesin",
    # thyroid
    "synthroid": "levothyroxine", "levoxyl": "levothyroxine",
    "unithroid": "levothyroxine", "tirosint": "levothyroxine",
    "armour thyroid": "thyroid desiccated",
    "cytomel": "liothyronine",
    # psychiatric/neuro
    "prozac": "fluoxetine", "sarafem": "fluoxetine",
    "zoloft": "sertraline", "paxil": "paroxetine",
    "celexa": "citalopram", "lexapro": "escitalopram",
    "luvox": "fluvoxamine", "effexor xr": "venlafaxine",
    "effexor": "venlafaxine", "cymbalta": "duloxetine",
    "pristiq": "desvenlafaxine", "fetzima": "levomilnacipran",
    "wellbutrin sr": "bupropion", "wellbutrin xl": "bupropion",
    "wellbutrin": "bupropion", "zyban": "bupropion",
    "remeron": "mirtazapine", "desyrel": "trazodone",
    "elavil": "amitriptyline", "pamelor": "nortriptyline",
    "tofranil": "imipramine", "norpramin": "desipramine",
    "anafranil": "clomipramine", "vivactil": "protriptyline",
    "nardil": "phenelzine", "parnate": "tranylcypromine",
    "emsam": "selegiline",
    "ativan": "lorazepam", "xanax": "alprazolam",
    "valium": "diazepam", "klonopin": "clonazepam",
    "tranxene": "clorazepate", "librium": "chlordiazepoxide",
    "restoril": "temazepam", "dalmane": "flurazepam",
    "halcion": "triazolam",
    "ambien": "zolpidem", "lunesta": "eszopiclone",
    "sonata": "zaleplon", "belsomra": "suvorexant",
    "abilify": "aripiprazole", "seroquel": "quetiapine",
    "seroquel xr": "quetiapine", "zyprexa": "olanzapine",
    "risperdal": "risperidone", "geodon": "ziprasidone",
    "latuda": "lurasidone", "rexulti": "brexpiprazole",
    "invega": "paliperidone", "fanapt": "iloperidone",
    "haldol": "haloperidol", "thorazine": "chlorpromazine",
    "mellaril": "thioridazine", "prolixin": "fluphenazine",
    "navane": "thiothixene", "trilafon": "perphenazine",
    "concerta": "methylphenidate", "ritalin": "methylphenidate",
    "adderall xr": "amphetamine salts", "adderall": "amphetamine salts",
    "vyvanse": "lisdexamfetamine", "strattera": "atomoxetine",
    "intuniv": "guanfacine", "kapvay": "clonidine",
    "neurontin": "gabapentin", "lyrica": "pregabalin",
    "topamax": "topiramate", "depakote er": "valproic acid",
    "depakote": "valproic acid", "depakene": "valproic acid",
    "lamictal": "lamotrigine", "tegretol": "carbamazepine",
    "trileptal": "oxcarbazepine", "keppra": "levetiracetam",
    "dilantin": "phenytoin", "phenobarbital": "phenobarbital",
    "zonegran": "zonisamide", "gabitril": "tiagabine",
    "sabril": "vigabatrin", "onfi": "clobazam",
    # cardiovascular
    "plavix": "clopidogrel", "effient": "prasugrel",
    "brilinta": "ticagrelor", "aggrenox": "aspirin-dipyridamole",
    "pletal": "cilostazol", "trental": "pentoxifylline",
    "ranexa": "ranolazine", "imdur": "isosorbide mononitrate",
    "isordil": "isosorbide dinitrate",
    "lanoxin": "digoxin", "digitek": "digoxin",
    "cordarone": "amiodarone", "pacerone": "amiodarone",
    "multaq": "dronedarone", "tikosyn": "dofetilide",
    "rythmol": "propafenone", "mexitil": "mexiletine",
    "betapace": "sotalol", "tambocor": "flecainide",
    # pain/rheumatology
    "celebrex": "celecoxib", "mobic": "meloxicam",
    "feldene": "piroxicam", "relafen": "nabumetone",
    "voltaren": "diclofenac", "daypro": "oxaprozin",
    "ansaid": "flurbiprofen", "lodine": "etodolac",
    "orudis": "ketoprofen",
    # allergy/antihistamines
    "benadryl": "diphenhydramine", "unisom": "diphenhydramine",
    "zyrtec": "cetirizine", "claritin": "loratadine",
    "clarinex": "desloratadine", "allegra": "fexofenadine",
    "xyzal": "levocetirizine", "astelin": "azelastine",
    # erectile dysfunction / urology
    "viagra": "sildenafil", "revatio": "sildenafil",
    "cialis": "tadalafil", "adcirca": "tadalafil",
    "levitra": "vardenafil", "stendra": "avanafil",
    "flomax": "tamsulosin", "rapaflo": "silodosin",
    "uroxatral": "alfuzosin", "cardura": "doxazosin",
    "hytrin": "terazosin", "proscar": "finasteride",
    "avodart": "dutasteride",
    # other common
    "laxative (bisacodyl)": "bisacodyl",
    "dulcolax (bisacodyl)": "bisacodyl",
    "allergy relief (fluticasone)": "fluticasone",
    "allergy relief (loratadine)": "loratadine",
    "allergy relief (cetirizine)": "cetirizine",
    "allergy relief (fexofenadine)": "fexofenadine",
    "allergy relief (levocetirizine)": "levocetirizine",
    "children's flonase allergy rlf": "fluticasone",
    "children's loratadine": "loratadine",
    "topamax": "topiramate",
    "seroquel": "quetiapine",
    "azilect": "rasagiline",
}

# ── 4. Salt forms to strip ────────────────────────────────────────────────────
SALT_SUFFIXES = re.compile(
    r'\s+\b(hcl|hydrochloride|hbr|hydrobromide|sodium|potassium|sulfate|'
    r'phosphate|citrate|maleate|tartrate|acetate|gluconate|lactate|'
    r'mesylate|besylate|tosylate|fumarate|succinate|oxalate|nitrate|'
    r'chloride|bromide|iodide|stearate|monohydrate|dihydrate|trihydrate|'
    r'anhydrous|monosodium|disodium|trisodium)\b',
    re.I
)

# ── 5. Strip these suffixes/descriptors ───────────────────────────────────────
DESCRIPTOR_PATTERNS = [
    re.compile(r'\s*\((?:bulk|pf|preservative.free|with sugar|sugar free|'
               r'anti.rheumatic|antihypertensive|porcine|human recombinant|'
               r'recombinant|disposable|emollient)\)', re.I),
    re.compile(r'\s*\bin (dextrose|nacl|saline|ns|d5w|water|0\.9%.*)\b.*$', re.I),
    re.compile(r'\s*\b(extra strength|maximum strength|max strength|ultra|'
               r'regular strength|arthritis pain|pm\b|nighttime|daytime|'
               r'24.?hr|24.?hour|once.daily|ec\b|enteric coated|enteric|'
               r'low dose|low strength|adult low dose|baby|children\'s|'
               r'infant|junior|pediatric|adult|senior|women\'s|men\'s|'
               r'advanced|plus\b|original)\b', re.I),
    re.compile(r'\s*(hfa|dpi)\b', re.I),
]

# ── 6. Parenthetical: extract active ingredient ───────────────────────────────
PAREN_EXTRACT = re.compile(r'^.*?\(([^)]+)\)\s*$')
KNOWN_GENERICS = {
    "acetaminophen", "bisacodyl", "fluticasone", "loratadine", "cetirizine",
    "fexofenadine", "levocetirizine", "senna", "ibuprofen", "naproxen",
    "aspirin", "pseudoephedrine", "diphenhydramine", "dextromethorphan",
    "guaifenesin", "calcium carbonate", "magnesium hydroxide",
    "aluminum hydroxide", "simethicone", "loperamide",
}

# ── 7. Vitamin name standardization ──────────────────────────────────────────
VITAMIN_MAP = {
    "vitamin b-12": "cyanocobalamin", "vitamin b12": "cyanocobalamin",
    "b-12": "cyanocobalamin", "b12": "cyanocobalamin",
    "vitamin b-1": "thiamine", "vitamin b1": "thiamine",
    "vitamin b-2": "riboflavin", "vitamin b2": "riboflavin",
    "vitamin b-6": "pyridoxine", "vitamin b6": "pyridoxine",
    "vitamin b-3": "niacin", "vitamin b3": "niacin",
    "vitamin d3": "cholecalciferol", "vitamin d-3": "cholecalciferol",
    "vitamin d2": "ergocalciferol", "vitamin d-2": "ergocalciferol",
    "vitamin c": "ascorbic acid",
    "vitamin e": "tocopherol",
    "vitamin a": "retinol",
    "vitamin k": "phytonadione",
    "folic acid": "folic acid",
    "coq10": "coenzyme q10", "co q-10": "coenzyme q10",
    "co q10": "coenzyme q10",
}

def _is_device_or_generic(name: str) -> bool:
    n = name.lower()
    if n in GENERIC_DESCRIPTORS:
        return True
    for kw in DEVICE_KEYWORDS:
        if kw in n:
            return True
    return False

def standardize_med(name: str):
    n = name.strip().lower()

    # skip * prefix
    if n.startswith("*"):
        return n

    # drop devices/generic descriptors
    if _is_device_or_generic(n):
        return None

    # vitamin map (before brand map)
    if n in VITAMIN_MAP:
        return VITAMIN_MAP[n]

    # brand → generic (exact match first)
    if n in BRAND_TO_GENERIC:
        return BRAND_TO_GENERIC[n]

    # parenthetical: if inner text is a known generic, use it
    paren_match = PAREN_EXTRACT.match(n)
    if paren_match:
        inner = paren_match.group(1).strip().lower()
        if inner in KNOWN_GENERICS:
            return inner
        # otherwise strip the parenthetical
        n = re.sub(r'\s*\(.*?\)', '', n).strip()

    # strip descriptor patterns
    for pat in DESCRIPTOR_PATTERNS:
        n = pat.sub('', n).strip()

    # strip salt suffixes (only if something remains)
    stripped = SALT_SUFFIXES.sub('', n).strip().rstrip(',').strip()
    if stripped:
        n = stripped

    # brand → generic again after cleaning
    if n in BRAND_TO_GENERIC:
        return BRAND_TO_GENERIC[n]

    return n if n else None


def clean_med_list(val: str) -> str:
    if not val or val == "-":
        return "-"
    meds = [standardize_med(m.strip()) for m in str(val).split(";")]
    meds = [m for m in meds if m]
    meds = list(dict.fromkeys(meds))  # remove duplicates, preserve order
    return "; ".join(meds) if meds else "-"
