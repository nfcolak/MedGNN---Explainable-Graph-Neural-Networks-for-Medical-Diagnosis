export interface DiagnosisInfo {
  description: string;
  mergedFrom?: string[];
}

const diagnosisInfo: Record<string, DiagnosisInfo> = {
  'Acute kidney failure': {
    description:
      'A sudden decline in kidney function that can cause waste products and fluid to build up quickly.',
  },
  'Acute pharyngitis': {
    description:
      'Inflammation or infection of the throat, often presenting with sore throat, fever, or painful swallowing.',
  },
  'Acute upper respiratory infection': {
    description:
      'A short-term infection of the nose, throat, sinuses, or upper airways, often similar to a common cold.',
  },
  'Alcohol abuse with intoxication': {
    description:
      'Harmful alcohol use with acute intoxication, which can affect mental status, coordination, and vital signs.',
  },
  Anemia: {
    description:
      'A condition with reduced red blood cells or hemoglobin, limiting oxygen delivery to body tissues.',
  },
  'Anxiety disorder': {
    description:
      'A mental health condition marked by excessive fear, worry, panic, or physical symptoms of anxiety.',
  },
  'Back or spine pain': {
    description:
      'Pain involving the back, neck, or spine, ranging from muscle strain to nerve, bone, or disc problems.',
    mergedFrom: [
      'Low back pain',
      'Strain of muscle',
      'Sprain of ligaments of lumbar spine',
      'Contusion of lower back and pelvis',
      'Unspecified lumbar vertebra fracture',
      'Neck joint or ligament sprain',
    ],
  },
  'Cardiovascular risk factor': {
    description:
      'Clinical factors such as hypertension, high cholesterol, diabetes, or smoking that increase heart disease risk.',
    mergedFrom: [
      'Essential hypertension',
      'Familial hypercholesterolemia',
      'Atherosclerotic coronary artery disease without angina',
    ],
  },
  Dehydration: {
    description:
      'A fluid deficit that can cause weakness, dizziness, abnormal labs, low blood pressure, or fast heart rate.',
  },
  'Diabetes mellitus': {
    description:
      'A chronic metabolic disease characterized by high blood glucose due to impaired insulin production or action.',
    mergedFrom: [
      'Type 1 diabetes mellitus without complications',
      'Type 2 diabetes mellitus without complications',
    ],
  },
  'End stage renal disease': {
    description:
      'Advanced chronic kidney failure in which the kidneys can no longer adequately support normal body function.',
  },
  Epilepsy: {
    description:
      'A neurologic disorder involving recurrent seizures caused by abnormal electrical activity in the brain.',
  },
  'GI bleed': {
    description:
      'Bleeding from the gastrointestinal tract, which may appear as vomiting blood, black stool, anemia, or weakness.',
    mergedFrom: ['Gastrointestinal hemorrhage', 'Hemorrhage of anus and rectum'],
  },
  'Head injury': {
    description:
      'Trauma to the scalp, skull, face, or brain, ranging from minor contusion to fractures or intracranial bleeding.',
    mergedFrom: [
      'Contusion of unspecified part of head',
      'Other specified injuries of head',
      'Open wound of other part of head',
      'Traumatic subdural hemorrhage without loss of consciousness',
      'Fracture of nasal bones',
    ],
  },
  'Heart failure': {
    description:
      'A condition where the heart cannot pump blood effectively, often causing shortness of breath or fluid overload.',
  },
  Hypokalemia: {
    description:
      'A low blood potassium level that may cause weakness, cramps, abnormal heart rhythms, or ECG changes.',
  },
  Hypotension: {
    description:
      'Low blood pressure, which may reduce organ perfusion and cause dizziness, weakness, or shock-like symptoms.',
  },
  Hypothyroidism: {
    description:
      'Low thyroid hormone activity, which can slow metabolism and cause fatigue, cold intolerance, or weight gain.',
  },
  'Laceration w/o fb of l idx fngr w/o damage to nail': {
    description:
      'A cut on the left index finger without a retained foreign body and without injury to the nail.',
  },
  'Limb injury or pain': {
    description:
      'Pain, swelling, or trauma involving an arm or leg, including sprains, fractures, wounds, or soft-tissue injury.',
    mergedFrom: [
      'Pain in unspecified limb',
      'Pain in unspecified knee',
      'Abrasion',
      'Abrasion of unspecified hand',
      'Displaced fifth metatarsal fracture',
      'Contusion of unspecified hip',
      'Distal radius fracture',
      'Right ankle ligament sprain',
    ],
  },
  'Lower respiratory disease': {
    description:
      'Disease affecting the lower airways or lungs, often causing cough, wheeze, fever, or shortness of breath.',
    mergedFrom: ['COPD with acute exacerbation', 'Pneumonia'],
  },
  'Major depressive disorder': {
    description:
      'A mood disorder with persistent low mood, loss of interest, and possible changes in sleep, appetite, or energy.',
  },
  'Multiple fractures of ribs': {
    description:
      'Two or more broken ribs, usually from trauma, which can cause chest pain and breathing difficulty.',
  },
  'Non-ST elevation (NSTEMI) myocardial infarction': {
    description:
      'A type of heart attack caused by reduced blood flow to heart muscle, often detected by troponin elevation.',
  },
  Sepsis: {
    description:
      'A severe body-wide response to infection that can lead to organ dysfunction and unstable vital signs.',
  },
  'Skin or soft-tissue infection': {
    description:
      'An infection of the skin or underlying tissues, such as cellulitis or abscess, often with redness or swelling.',
    mergedFrom: [
      'Cellulitis of unspecified part of limb',
      'Cutaneous abscess of buttock',
      'Local skin and subcutaneous tissue infection',
      'Infection following a procedure',
    ],
  },
  'UTI or pyelonephritis': {
    description:
      'Infection of the urinary tract or kidneys, often causing urinary symptoms, fever, flank pain, or abnormal urine tests.',
    mergedFrom: ['Urinary tract infection', 'Acute pyelonephritis', 'Tubulo-interstitial nephritis'],
  },
  'Unsp intestnl obst': {
    description:
      'An intestinal obstruction that blocks normal bowel passage and may cause abdominal pain, distention, or vomiting.',
  },
  'Unspecified asthma with (acute) exacerbation': {
    description:
      'An asthma flare with worsened airway narrowing, wheezing, cough, or shortness of breath.',
  },
  'Unspecified atrial fibrillation': {
    description:
      'An irregular heart rhythm from the atria that can cause palpitations, fast heart rate, or stroke risk.',
  },
};

const fallbackDiagnosisInfo: DiagnosisInfo = {
  description: 'A clinical diagnosis category assigned to this ED encounter in the ProtGNN dataset.',
};

export function getDiagnosisInfo(diagnosis?: string | null): DiagnosisInfo {
  if (!diagnosis) {
    return { description: 'Select a patient graph to view the diagnosis description.' };
  }
  return diagnosisInfo[diagnosis] ?? fallbackDiagnosisInfo;
}
