export interface MedicationInfo {
  className: string;
  description: string;
}

const medicationInfo: Record<string, MedicationInfo> = {
  acetaminophen: {
    className: 'Analgesic / antipyretic',
    description: 'Used to reduce pain and fever; often recorded for headache, injury pain, or general discomfort.',
  },
  acyclovir: {
    className: 'Antiviral',
    description: 'Used for herpes-family viral infections, including shingles, genital herpes, and severe cold sores.',
  },
  albuterol: {
    className: 'Short-acting bronchodilator',
    description: 'Relaxes airway smooth muscle and is commonly used for wheezing, asthma, or COPD flares.',
  },
  allopurinol: {
    className: 'Urate-lowering therapy',
    description: 'Lowers uric acid over time and is commonly used in patients with gout history.',
  },
  alprazolam: {
    className: 'Benzodiazepine',
    description: 'A sedating anti-anxiety medication that may be used for panic or anxiety symptoms.',
  },
  amitriptyline: {
    className: 'Tricyclic antidepressant',
    description: 'Used for depression, neuropathic pain, migraine prevention, or sleep-related symptoms.',
  },
  amlodipine: {
    className: 'Calcium-channel blocker',
    description: 'Used to treat high blood pressure or angina by relaxing blood vessels.',
  },
  ascorbic_acid: {
    className: 'Vitamin C supplement',
    description: 'A vitamin supplement used for nutritional support or deficiency prevention.',
  },
  aspirin: {
    className: 'Antiplatelet / analgesic',
    description: 'Used for pain or fever and, at low doses, to reduce platelet clotting in cardiovascular disease.',
  },
  atenolol: {
    className: 'Beta blocker',
    description: 'Slows heart rate and lowers blood pressure; used for hypertension, angina, or rhythm control.',
  },
  atorvastatin: {
    className: 'Statin',
    description: 'Lowers cholesterol and is used to reduce cardiovascular risk.',
  },
  bisacodyl: {
    className: 'Stimulant laxative',
    description: 'Used to treat constipation by stimulating bowel movement.',
  },
  blood_sugar_diagnostic: {
    className: 'Diabetes testing supply',
    description: 'Glucose test strips or related supplies used for blood sugar monitoring.',
  },
  budesonide_formoterol: {
    className: 'Inhaled steroid / long-acting bronchodilator',
    description: 'Maintenance inhaler used for asthma or COPD to reduce inflammation and improve airflow.',
  },
  bupropion: {
    className: 'Antidepressant',
    description: 'Used for depression, smoking cessation support, or related mood symptoms.',
  },
  calcium_carbonate: {
    className: 'Calcium supplement / antacid',
    description: 'Used for calcium supplementation or relief of acid-related stomach symptoms.',
  },
  calcium_carbonate_vitamin_d3: {
    className: 'Calcium + vitamin D supplement',
    description: 'Used for bone health support or calcium/vitamin D supplementation.',
  },
  carvedilol: {
    className: 'Beta blocker',
    description: 'Used for heart failure, high blood pressure, or cardiovascular protection.',
  },
  cetirizine: {
    className: 'Antihistamine',
    description: 'Used for allergy symptoms such as itching, sneezing, hives, or runny nose.',
  },
  chlorthalidone: {
    className: 'Thiazide-like diuretic',
    description: 'Used for high blood pressure and fluid management by increasing salt and water excretion.',
  },
  cholecalciferol: {
    className: 'Vitamin D3 supplement',
    description: 'Used to prevent or treat vitamin D deficiency and support bone health.',
  },
  citalopram: {
    className: 'SSRI antidepressant',
    description: 'Used for depression or anxiety by increasing serotonin signaling.',
  },
  clonazepam: {
    className: 'Benzodiazepine',
    description: 'A sedating medication used for seizures, panic symptoms, or anxiety-related conditions.',
  },
  clonidine: {
    className: 'Alpha-2 agonist',
    description: 'Used for high blood pressure and sometimes for withdrawal, anxiety, or autonomic symptoms.',
  },
  clopidogrel: {
    className: 'Antiplatelet',
    description: 'Helps prevent platelet clotting, often used after stroke, heart attack, or coronary stenting.',
  },
  cyanocobalamin: {
    className: 'Vitamin B12 supplement',
    description: 'Used to treat or prevent vitamin B12 deficiency and related anemia or neuropathy.',
  },
  cyclobenzaprine: {
    className: 'Muscle relaxant',
    description: 'Used short term for muscle spasm or musculoskeletal pain.',
  },
  diazepam: {
    className: 'Benzodiazepine',
    description: 'A sedating medication used for anxiety, muscle spasm, alcohol withdrawal, or seizures.',
  },
  diltiazem: {
    className: 'Calcium-channel blocker',
    description: 'Used for blood pressure control, angina, or certain fast heart rhythms.',
  },
  docusate: {
    className: 'Stool softener',
    description: 'Used to make bowel movements easier in constipation-prone patients.',
  },
  duloxetine: {
    className: 'SNRI antidepressant',
    description: 'Used for depression, anxiety, neuropathic pain, or chronic musculoskeletal pain.',
  },
  enoxaparin: {
    className: 'Low-molecular-weight heparin',
    description: 'An injectable anticoagulant used to prevent or treat blood clots.',
  },
  ergocalciferol: {
    className: 'Vitamin D2 supplement',
    description: 'Used for vitamin D deficiency replacement.',
  },
  escitalopram: {
    className: 'SSRI antidepressant',
    description: 'Used for depression or anxiety disorders by increasing serotonin signaling.',
  },
  ferrous: {
    className: 'Iron supplement',
    description: 'Used to treat or prevent iron deficiency, often in patients with anemia.',
  },
  finasteride: {
    className: '5-alpha-reductase inhibitor',
    description: 'Used for enlarged prostate symptoms or hair loss by altering androgen metabolism.',
  },
  fish_oil: {
    className: 'Omega-3 supplement',
    description: 'A dietary supplement sometimes used for triglyceride or cardiovascular support.',
  },
  fluoxetine: {
    className: 'SSRI antidepressant',
    description: 'Used for depression, anxiety, OCD, and related mood disorders.',
  },
  fluticasone: {
    className: 'Corticosteroid',
    description: 'Used as a nasal spray or inhaled steroid for allergy, asthma, or airway inflammation.',
  },
  fluticasone_salmeterol: {
    className: 'Inhaled steroid / long-acting bronchodilator',
    description: 'Maintenance inhaler used for asthma or COPD to reduce inflammation and improve airflow.',
  },
  folic_acid: {
    className: 'Folate supplement',
    description: 'Used to prevent or treat folate deficiency and support red blood cell production.',
  },
  furosemide: {
    className: 'Loop diuretic',
    description: 'Used to remove excess fluid in heart failure, kidney disease, or edema.',
  },
  gabapentin: {
    className: 'Anticonvulsant / neuropathic pain agent',
    description: 'Used for nerve pain, seizures, or related chronic pain syndromes.',
  },
  glipizide: {
    className: 'Sulfonylurea',
    description: 'Used in type 2 diabetes to increase insulin release and lower blood glucose.',
  },
  hydrochlorothiazide: {
    className: 'Thiazide diuretic',
    description: 'Used for high blood pressure and mild fluid control.',
  },
  hydrocortisone: {
    className: 'Corticosteroid',
    description: 'Used to reduce inflammation or replace steroid hormones in adrenal insufficiency.',
  },
  hydromorphone: {
    className: 'Opioid analgesic',
    description: 'A strong pain medication used for moderate to severe pain.',
  },
  ibuprofen: {
    className: 'NSAID',
    description: 'Used for pain, fever, and inflammation; common in musculoskeletal complaints.',
  },
  insulin_glargine: {
    className: 'Long-acting insulin',
    description: 'Provides basal insulin coverage for diabetes management.',
  },
  insulin_lispro: {
    className: 'Rapid-acting insulin',
    description: 'Used around meals or for correction of high blood glucose.',
  },
  isosorbide_mononitrate: {
    className: 'Nitrate vasodilator',
    description: 'Used to prevent angina by relaxing blood vessels and reducing cardiac workload.',
  },
  lactulose: {
    className: 'Osmotic laxative',
    description: 'Used for constipation or to reduce ammonia levels in hepatic encephalopathy.',
  },
  lamotrigine: {
    className: 'Anticonvulsant / mood stabilizer',
    description: 'Used for seizures or bipolar disorder maintenance.',
  },
  latanoprost: {
    className: 'Prostaglandin eye drop',
    description: 'Used to lower eye pressure in glaucoma or ocular hypertension.',
  },
  levetiracetam: {
    className: 'Anticonvulsant',
    description: 'Used to prevent or treat seizures.',
  },
  levothyroxine: {
    className: 'Thyroid hormone replacement',
    description: 'Used to treat hypothyroidism by replacing thyroid hormone.',
  },
  lidocaine: {
    className: 'Local anesthetic',
    description: 'Used to numb tissue or reduce localized pain.',
  },
  lisinopril: {
    className: 'ACE inhibitor',
    description: 'Used for high blood pressure, heart failure, and kidney protection in selected patients.',
  },
  loratadine: {
    className: 'Antihistamine',
    description: 'Used for allergy symptoms such as sneezing, itching, or runny nose.',
  },
  lorazepam: {
    className: 'Benzodiazepine',
    description: 'A sedating anti-anxiety medication also used for agitation, seizures, or alcohol withdrawal contexts.',
  },
  losartan: {
    className: 'Angiotensin receptor blocker',
    description: 'Used for high blood pressure, kidney protection, or heart-related indications.',
  },
  melatonin: {
    className: 'Sleep supplement',
    description: 'Used to support sleep timing or insomnia symptoms.',
  },
  metformin: {
    className: 'Biguanide diabetes medication',
    description: 'First-line type 2 diabetes medication that lowers liver glucose production and improves insulin sensitivity.',
  },
  metoprolol: {
    className: 'Beta blocker',
    description: 'Slows heart rate and lowers blood pressure; used for hypertension, angina, heart failure, or rhythm control.',
  },
  mirtazapine: {
    className: 'Antidepressant',
    description: 'Used for depression and sometimes for insomnia or appetite support.',
  },
  montelukast: {
    className: 'Leukotriene receptor antagonist',
    description: 'Used for asthma or allergy-related airway symptoms.',
  },
  multivitamin: {
    className: 'Vitamin supplement',
    description: 'Used for general nutritional supplementation.',
  },
  naproxen: {
    className: 'NSAID',
    description: 'Used for pain and inflammation, often in musculoskeletal conditions.',
  },
  nitroglycerin: {
    className: 'Nitrate vasodilator',
    description: 'Used for chest pain due to angina by dilating blood vessels.',
  },
  omeprazole: {
    className: 'Proton pump inhibitor',
    description: 'Reduces stomach acid and is used for reflux, ulcers, or gastritis-related symptoms.',
  },
  ondansetron: {
    className: 'Antiemetic',
    description: 'Used to treat nausea and vomiting by blocking serotonin receptors involved in emesis.',
  },
  one_daily_multivitamin: {
    className: 'Vitamin supplement',
    description: 'Used for general daily nutritional supplementation.',
  },
  oxycodone: {
    className: 'Opioid analgesic',
    description: 'Used for moderate to severe pain.',
  },
  oxycodone_acetaminophen: {
    className: 'Opioid / analgesic combination',
    description: 'Combination pain medication containing oxycodone and acetaminophen.',
  },
  pantoprazole: {
    className: 'Proton pump inhibitor',
    description: 'Reduces stomach acid and is used for reflux, ulcers, or GI bleed prophylaxis contexts.',
  },
  paroxetine: {
    className: 'SSRI antidepressant',
    description: 'Used for depression, anxiety, panic disorder, and related conditions.',
  },
  polyethylene_glycol_3350: {
    className: 'Osmotic laxative',
    description: 'Used for constipation by drawing water into the bowel.',
  },
  potassium: {
    className: 'Electrolyte supplement',
    description: 'Used to replace or prevent low potassium levels.',
  },
  pravastatin: {
    className: 'Statin',
    description: 'Lowers cholesterol and is used to reduce cardiovascular risk.',
  },
  prednisone: {
    className: 'Systemic corticosteroid',
    description: 'Used to reduce inflammation in asthma/COPD flares, allergic reactions, and autoimmune conditions.',
  },
  prochlorperazine: {
    className: 'Antiemetic / antipsychotic',
    description: 'Used for nausea, vomiting, migraine-associated symptoms, or agitation in some contexts.',
  },
  quetiapine: {
    className: 'Atypical antipsychotic',
    description: 'Used for bipolar disorder, schizophrenia, depression augmentation, or severe agitation contexts.',
  },
  ranitidine: {
    className: 'H2 blocker',
    description: 'Acid-reducing medication historically used for reflux or ulcers.',
  },
  rosuvastatin: {
    className: 'Statin',
    description: 'Lowers cholesterol and is used to reduce cardiovascular risk.',
  },
  sennosides: {
    className: 'Stimulant laxative',
    description: 'Used to treat constipation by stimulating bowel movement.',
  },
  sertraline: {
    className: 'SSRI antidepressant',
    description: 'Used for depression, anxiety, PTSD, and related mood disorders.',
  },
  sildenafil: {
    className: 'PDE-5 inhibitor',
    description: 'Used for erectile dysfunction or pulmonary arterial hypertension depending on formulation and context.',
  },
  simvastatin: {
    className: 'Statin',
    description: 'Lowers cholesterol and is used to reduce cardiovascular risk.',
  },
  spironolactone: {
    className: 'Potassium-sparing diuretic',
    description: 'Used for heart failure, resistant hypertension, ascites, or hormonal indications.',
  },
  sulfamethoxazole_trimethoprim: {
    className: 'Antibiotic',
    description: 'Used for selected bacterial infections, including some urinary, skin, and respiratory infections.',
  },
  tamsulosin: {
    className: 'Alpha-1 blocker',
    description: 'Used for urinary symptoms from enlarged prostate by relaxing smooth muscle in the urinary tract.',
  },
  thiamine: {
    className: 'Vitamin B1 supplement',
    description: 'Used to prevent or treat thiamine deficiency, especially in alcohol-use or malnutrition contexts.',
  },
  topiramate: {
    className: 'Anticonvulsant',
    description: 'Used for seizures, migraine prevention, and selected neurologic or psychiatric indications.',
  },
  torsemide: {
    className: 'Loop diuretic',
    description: 'Used to remove excess fluid in heart failure, kidney disease, or edema.',
  },
  tramadol: {
    className: 'Opioid-like analgesic',
    description: 'Used for moderate pain, with opioid and serotonin/norepinephrine effects.',
  },
  trazodone: {
    className: 'Antidepressant / sleep agent',
    description: 'Used for depression and commonly for insomnia symptoms.',
  },
  triamcinolone_acetonide: {
    className: 'Corticosteroid',
    description: 'Used to reduce inflammation in skin, joint, nasal, or allergy-related conditions depending on formulation.',
  },
  valsartan: {
    className: 'Angiotensin receptor blocker',
    description: 'Used for high blood pressure, heart failure, or kidney/cardiovascular protection.',
  },
  venlafaxine: {
    className: 'SNRI antidepressant',
    description: 'Used for depression, anxiety, panic disorder, or neuropathic pain contexts.',
  },
  warfarin: {
    className: 'Vitamin K antagonist anticoagulant',
    description: 'Used to prevent or treat blood clots, with monitoring of anticoagulation effect.',
  },
  zolpidem: {
    className: 'Sedative-hypnotic',
    description: 'Used short term for insomnia.',
  },
};

function normalizeMedicationName(value?: unknown) {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/^(med_|pyx_)/, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

export function getMedicationInfo(label?: unknown, originalFeatureName?: unknown): MedicationInfo {
  const keys = [normalizeMedicationName(originalFeatureName), normalizeMedicationName(label)];
  for (const key of keys) {
    if (key && medicationInfo[key]) return medicationInfo[key];
  }
  return {
    className: 'Medication',
    description:
      'Medication feature recorded in this patient graph. A curated short description is not available for this item yet.',
  };
}
