import type { PatientGraphNode } from '../types/graph';

export interface ChiefComplaintInfo {
  displayName: string;
  description: string;
}

const complaintInfo: Record<string, ChiefComplaintInfo> = {
  abdominal_distention: {
    displayName: 'Abdominal distention',
    description: 'Visible or felt swelling of the abdomen. It can occur with gas, fluid, constipation, bowel obstruction, liver disease, or inflammation inside the abdomen.',
  },
  abdominal_pain: {
    displayName: 'Abdominal pain',
    description: 'Pain felt in the belly. Possible sources include the stomach, intestines, liver, gallbladder, pancreas, urinary tract, reproductive organs, or abdominal blood vessels.',
  },
  abnormal_imaging: {
    displayName: 'Abnormal imaging',
    description: 'A concerning finding on X-ray, CT, ultrasound, MRI, or another imaging study that prompted emergency evaluation.',
  },
  abnormal_labs: {
    displayName: 'Abnormal labs',
    description: 'Blood or urine results outside the expected range, such as electrolyte problems, kidney markers, blood counts, liver tests, or infection-related markers.',
  },
  abscess: {
    displayName: 'Abscess',
    description: 'A collection of pus caused by infection, often producing localized pain, swelling, warmth, redness, and sometimes fever.',
  },
  alcohol_intoxication: {
    displayName: 'Alcohol intoxication',
    description: 'Acute effects of alcohol on the brain and body, including impaired coordination, slowed thinking, vomiting, low blood sugar, injury risk, or depressed breathing in severe cases.',
  },
  altered_mental_status: {
    displayName: 'Altered mental status',
    description: 'A change in alertness, orientation, behavior, or thinking. Causes can include infection, stroke, seizure, low oxygen, intoxication, medications, or metabolic problems.',
  },
  anemia: {
    displayName: 'Anemia',
    description: 'Low red blood cell or hemoglobin level, which can reduce oxygen delivery and cause fatigue, weakness, dizziness, shortness of breath, or chest discomfort.',
  },
  ankle_pain: {
    displayName: 'Ankle pain',
    description: 'Pain around the ankle joint, commonly related to sprain, fracture, tendon injury, arthritis, infection, gout, or swelling after trauma.',
  },
  anxiety: {
    displayName: 'Anxiety',
    description: 'Intense worry or panic that can produce chest tightness, palpitations, shortness of breath, trembling, nausea, or a sense of danger.',
  },
  arm_pain: {
    displayName: 'Arm pain',
    description: 'Pain in the upper limb. It may come from muscle, tendon, joint, bone, nerve, blood vessel, or referred pain from the neck or heart.',
  },
  arm_swelling: {
    displayName: 'Arm swelling',
    description: 'Enlargement of the arm from fluid, inflammation, infection, injury, poor venous drainage, or a possible blood clot.',
  },
  assault: {
    displayName: 'Assault',
    description: 'Injury or symptoms after physical violence. Evaluation often focuses on trauma patterns, head injury, fractures, wounds, pain, and safety concerns.',
  },
  asthma_exacerbation: {
    displayName: 'Asthma exacerbation',
    description: 'A flare of airway narrowing and inflammation, often causing wheezing, cough, chest tightness, and difficulty breathing.',
  },
  atrial_fibrillation: {
    displayName: 'Atrial fibrillation',
    description: 'An irregular heart rhythm from the upper chambers of the heart, which can cause palpitations, fast heart rate, shortness of breath, fatigue, or stroke risk.',
  },
  back_pain: {
    displayName: 'Back pain',
    description: 'Pain in the back from muscle strain, spine joints, discs, nerves, trauma, infection, kidney problems, or less common vascular causes.',
  },
  bowel_obstruction: {
    displayName: 'Bowel obstruction',
    description: 'Blocked movement through the intestines, often causing abdominal pain, bloating, vomiting, constipation, and inability to pass gas.',
  },
  calf_pain: {
    displayName: 'Calf pain',
    description: 'Pain in the lower leg that can arise from muscle strain, cramp, tendon injury, poor circulation, infection, or a possible blood clot.',
  },
  chest_pain: {
    displayName: 'Chest pain',
    description: 'Pain or pressure in the chest. Important causes include heart attack, lung disease, blood clots, reflux, muscle strain, infection, and anxiety.',
  },
  constipation: {
    displayName: 'Constipation',
    description: 'Difficult, infrequent, or incomplete bowel movements, sometimes causing abdominal pain, bloating, nausea, or rectal discomfort.',
  },
  cough: {
    displayName: 'Cough',
    description: 'A reflex that clears the airways. It can occur with viral illness, pneumonia, asthma, COPD, reflux, heart failure, or airway irritation.',
  },
  depression: {
    displayName: 'Depression',
    description: 'Low mood or loss of interest that may be accompanied by sleep changes, appetite changes, low energy, impaired concentration, or thoughts of self-harm.',
  },
  diarrhea: {
    displayName: 'Diarrhea',
    description: 'Loose or frequent stools that may result from infection, inflammation, medication effects, food intolerance, or bowel disease.',
  },
  dizziness: {
    displayName: 'Dizziness',
    description: 'A sense of lightheadedness, imbalance, spinning, or near-fainting. Causes include dehydration, inner ear disease, low blood pressure, rhythm problems, or neurologic disease.',
  },
  dyspnea: {
    displayName: 'Dyspnea',
    description: 'Shortness of breath or uncomfortable breathing. It can reflect lung disease, heart failure, anemia, infection, blood clot, anxiety, or metabolic stress.',
  },
  dysuria: {
    displayName: 'Dysuria',
    description: 'Pain or burning with urination, commonly associated with urinary tract infection, urethral irritation, stones, or sexually transmitted infection.',
  },
  ear_pain: {
    displayName: 'Ear pain',
    description: 'Pain in or around the ear, often related to ear infection, fluid behind the eardrum, ear canal inflammation, dental disease, or jaw joint pain.',
  },
  elbow_pain: {
    displayName: 'Elbow pain',
    description: 'Pain around the elbow from tendon irritation, fracture, dislocation, bursitis, arthritis, nerve irritation, or trauma.',
  },
  epistaxis: {
    displayName: 'Epistaxis',
    description: 'Nosebleed, usually from fragile nasal blood vessels but sometimes associated with trauma, dryness, high blood pressure, or blood-thinning medication.',
  },
  extremity_injury: {
    displayName: 'Extremity injury',
    description: 'Trauma involving an arm or leg, including sprain, fracture, contusion, dislocation, wound, tendon injury, or neurovascular concern.',
  },
  extremity_swelling: {
    displayName: 'Extremity swelling',
    description: 'Swelling in an arm or leg from fluid buildup, inflammation, injury, infection, venous disease, lymphatic problems, or a possible clot.',
  },
  eye_pain: {
    displayName: 'Eye pain',
    description: 'Pain in or around the eye, which can occur with corneal injury, infection, inflammation, glaucoma, headache syndromes, or trauma.',
  },
  facial_swelling: {
    displayName: 'Facial swelling',
    description: 'Swelling of the face from infection, allergic reaction, dental disease, trauma, salivary gland disease, or fluid retention.',
  },
  fall: {
    displayName: 'Fall',
    description: 'A fall can cause head injury, fracture, soft-tissue injury, bleeding, or pain, and may also signal dizziness, weakness, infection, or medication effects.',
  },
  fatigue: {
    displayName: 'Fatigue',
    description: 'Unusual tiredness or low energy. Causes include infection, anemia, dehydration, sleep problems, heart disease, endocrine disease, medications, or mood disorders.',
  },
  fever: {
    displayName: 'Fever',
    description: 'Elevated body temperature, most often reflecting infection or inflammation, but also possible with heat illness, medications, clots, or autoimmune disease.',
  },
  flank_pain: {
    displayName: 'Flank pain',
    description: 'Pain along the side of the back or abdomen, often associated with kidney stones, kidney infection, muscle strain, or abdominal disease.',
  },
  focal_neuro_deficit: {
    displayName: 'Focal neurologic deficit',
    description: 'Loss or change in a specific neurologic function, such as weakness, numbness, speech difficulty, facial droop, or vision change, raising concern for brain, spine, or nerve disease.',
  },
  foot_pain: {
    displayName: 'Foot pain',
    description: 'Pain in the foot from sprain, fracture, tendon injury, plantar fascia irritation, arthritis, infection, neuropathy, or poor circulation.',
  },
  gait_disturbance: {
    displayName: 'Gait disturbance',
    description: 'A change in walking or balance that may reflect pain, weakness, dizziness, neurologic disease, intoxication, or musculoskeletal injury.',
  },
  generalized_pain: {
    displayName: 'Generalized pain',
    description: 'Pain felt across multiple body areas. It may occur with infection, inflammation, trauma, medication effects, chronic pain syndromes, or metabolic illness.',
  },
  gi_bleed: {
    displayName: 'GI bleed',
    description: 'Bleeding from the digestive tract, which may appear as vomiting blood, black stool, red blood per rectum, anemia, weakness, or dizziness.',
  },
  hand_laceration: {
    displayName: 'Hand laceration',
    description: 'A cut on the hand, where assessment often considers bleeding, tendon movement, nerve sensation, contamination, and retained foreign material.',
  },
  hand_pain: {
    displayName: 'Hand pain',
    description: 'Pain in the hand from injury, fracture, tendon disease, arthritis, infection, nerve compression, or inflammation.',
  },
  head_injury: {
    displayName: 'Head injury',
    description: 'Trauma to the scalp, skull, face, or brain. Key concerns include concussion, bleeding, fracture, vomiting, confusion, severe headache, and neurologic changes.',
  },
  headache: {
    displayName: 'Headache',
    description: 'Pain in the head or face. Causes range from migraine and tension headache to infection, bleeding, high pressure, vascular disease, or medication effects.',
  },
  hematuria: {
    displayName: 'Hematuria',
    description: 'Blood in the urine, which may occur with urinary tract infection, kidney stone, trauma, kidney disease, prostate disease, or urinary tract tumors.',
  },
  hip_pain: {
    displayName: 'Hip pain',
    description: 'Pain around the hip from fracture, arthritis, bursitis, tendon injury, infection, referred back pain, or trauma.',
  },
  hyperglycemia: {
    displayName: 'Hyperglycemia',
    description: 'High blood glucose, commonly related to diabetes, infection, missed insulin or medication, dehydration, steroid use, or physiologic stress.',
  },
  hyperkalemia: {
    displayName: 'Hyperkalemia',
    description: 'High blood potassium, which can occur with kidney dysfunction, medications, tissue breakdown, acidosis, or excess potassium intake and may affect heart rhythm.',
  },
  hypertension: {
    displayName: 'Hypertension',
    description: 'High blood pressure. In the emergency setting, concern increases when it is associated with chest pain, neurologic symptoms, kidney injury, or heart strain.',
  },
  hypoglycemia: {
    displayName: 'Hypoglycemia',
    description: 'Low blood glucose, which can cause sweating, shaking, confusion, weakness, seizure, or loss of consciousness.',
  },
  hypotension: {
    displayName: 'Hypotension',
    description: 'Low blood pressure that may reduce blood flow to organs. It can occur with dehydration, bleeding, infection, heart problems, allergic reaction, or medications.',
  },
  hypoxia: {
    displayName: 'Hypoxia',
    description: 'Insufficient oxygen reaching body tissues, often reflected by low oxygen saturation and associated with lung disease, heart disease, anemia, shock, or airway problems.',
  },
  influenza_like_illness: {
    displayName: 'Influenza-like illness',
    description: 'A syndrome of fever, cough, sore throat, body aches, headache, chills, or fatigue, often caused by influenza or other respiratory viruses.',
  },
  knee_pain: {
    displayName: 'Knee pain',
    description: 'Pain around the knee from sprain, meniscus injury, fracture, arthritis, bursitis, tendon injury, infection, gout, or trauma.',
  },
  laceration: {
    displayName: 'Laceration',
    description: 'A cut or tear in the skin. Evaluation considers depth, bleeding, contamination, tendon or nerve involvement, and whether a foreign body is present.',
  },
  leg_pain: {
    displayName: 'Leg pain',
    description: 'Pain in the lower limb from muscle strain, joint disease, fracture, nerve irritation, poor circulation, infection, or a possible blood clot.',
  },
  motor_vehicle_collision: {
    displayName: 'Motor vehicle collision',
    description: 'Symptoms after a traffic crash. Common concerns include head, neck, chest, abdominal, spine, and extremity injuries.',
  },
  nausea: {
    displayName: 'Nausea',
    description: 'The feeling of needing to vomit. It can occur with gastrointestinal illness, pregnancy, pain, medications, infection, dizziness, metabolic disease, or heart disease.',
  },
  neck_pain: {
    displayName: 'Neck pain',
    description: 'Pain in the neck from muscle strain, spine disease, trauma, infection, nerve compression, or referred pain from nearby structures.',
  },
  palpitations: {
    displayName: 'Palpitations',
    description: 'Awareness of a fast, irregular, or forceful heartbeat. Causes include arrhythmias, anxiety, stimulants, thyroid disease, dehydration, or electrolyte problems.',
  },
  pneumonia: {
    displayName: 'Pneumonia',
    description: 'Infection of the lung tissue, often causing cough, fever, shortness of breath, chest discomfort, low oxygen, or abnormal lung sounds.',
  },
  presyncope: {
    displayName: 'Presyncope',
    description: 'A near-fainting sensation with lightheadedness or weakness, often related to low blood pressure, dehydration, rhythm problems, bleeding, or vasovagal episodes.',
  },
  psychiatric_evaluation: {
    displayName: 'Psychiatric evaluation',
    description: 'Assessment for acute mental health concerns such as severe mood symptoms, psychosis, agitation, safety risk, intoxication, or inability to care for oneself.',
  },
  rash: {
    displayName: 'Rash',
    description: 'A change in skin color, texture, or pattern caused by infection, allergy, inflammation, medication reaction, autoimmune disease, or irritation.',
  },
  rectal_bleeding: {
    displayName: 'Rectal bleeding',
    description: 'Blood passed from the rectum, ranging from hemorrhoids or fissure to diverticular bleeding, inflammation, infection, or upper gastrointestinal bleeding.',
  },
  rectal_pain: {
    displayName: 'Rectal pain',
    description: 'Pain around the rectum or anus, which can occur with hemorrhoids, fissure, abscess, infection, inflammation, trauma, or constipation.',
  },
  rib_pain: {
    displayName: 'Rib pain',
    description: 'Pain along the chest wall or ribs, often from contusion, fracture, muscle strain, inflammation, cough-related injury, or trauma.',
  },
  seizure: {
    displayName: 'Seizure',
    description: 'A burst of abnormal electrical activity in the brain that can cause shaking, staring, confusion, loss of awareness, or post-event sleepiness.',
  },
  shoulder_pain: {
    displayName: 'Shoulder pain',
    description: 'Pain around the shoulder from rotator cuff disease, dislocation, fracture, arthritis, referred neck pain, heart disease, or abdominal irritation.',
  },
  sore_throat: {
    displayName: 'Sore throat',
    description: 'Throat pain from viral infection, strep infection, tonsillitis, reflux, irritation, abscess, or other inflammation of the upper airway.',
  },
  subdural_hemorrhage: {
    displayName: 'Subdural hemorrhage',
    description: 'Bleeding between the brain surface and its outer covering, often after head trauma and more concerning with headache, confusion, weakness, vomiting, or anticoagulant use.',
  },
  substance_use: {
    displayName: 'Substance use',
    description: 'Health effects related to drugs or non-prescribed substances, including intoxication, withdrawal, overdose risk, injury, infection, or mental status changes.',
  },
  suicidal_ideation: {
    displayName: 'Suicidal ideation',
    description: 'Thoughts of self-harm or ending one’s life. Emergency evaluation focuses on immediate safety, intent, plan, supports, mental health symptoms, and substance use.',
  },
  syncope: {
    displayName: 'Syncope',
    description: 'A brief loss of consciousness from reduced blood flow to the brain, with causes including vasovagal episodes, dehydration, heart rhythm problems, bleeding, or seizure mimics.',
  },
  tachycardia: {
    displayName: 'Tachycardia',
    description: 'A faster-than-expected heart rate, which can occur with fever, pain, dehydration, anemia, anxiety, infection, blood loss, stimulant use, or arrhythmia.',
  },
  toe_pain: {
    displayName: 'Toe pain',
    description: 'Pain in a toe from fracture, sprain, nail disease, infection, gout, arthritis, or circulation problems.',
  },
  transfer: {
    displayName: 'Transfer',
    description: 'Arrival from another care setting for further evaluation, monitoring, imaging, specialty care, or admission-level treatment.',
  },
  urinary_frequency: {
    displayName: 'Urinary frequency',
    description: 'Urinating more often than usual, commonly associated with urinary tract infection, high glucose, diuretics, pregnancy, prostate symptoms, or bladder irritation.',
  },
  urinary_retention: {
    displayName: 'Urinary retention',
    description: 'Difficulty emptying the bladder, which can cause lower abdominal pain and may relate to prostate enlargement, medications, nerve problems, infection, or obstruction.',
  },
  vaginal_bleeding: {
    displayName: 'Vaginal bleeding',
    description: 'Bleeding from the vagina outside expected patterns or during pregnancy, with causes including hormonal changes, pregnancy complications, infection, trauma, or uterine disease.',
  },
  visual_changes: {
    displayName: 'Visual changes',
    description: 'New change in vision such as blurring, loss of vision, double vision, flashes, or field loss, which can arise from eye, nerve, brain, vascular, or migraine-related causes.',
  },
  vomiting: {
    displayName: 'Vomiting',
    description: 'Forceful emptying of stomach contents. Causes include gastrointestinal infection, obstruction, pregnancy, medications, intoxication, brain pressure, metabolic disease, or severe pain.',
  },
  weakness: {
    displayName: 'Weakness',
    description: 'Reduced strength or low energy. It may reflect neurologic disease, infection, dehydration, anemia, electrolyte problems, heart disease, or generalized illness.',
  },
  wound_evaluation: {
    displayName: 'Wound evaluation',
    description: 'Assessment of a cut, puncture, bite, burn, surgical wound, or ulcer for infection, depth, bleeding, foreign material, and tissue damage.',
  },
  wrist_injury: {
    displayName: 'Wrist injury',
    description: 'Trauma to the wrist, commonly evaluated for fracture, sprain, dislocation, tendon injury, nerve symptoms, and swelling.',
  },
  wrist_pain: {
    displayName: 'Wrist pain',
    description: 'Pain around the wrist from sprain, fracture, tendon inflammation, arthritis, nerve compression, infection, or overuse.',
  },
};

function normalizeComplaintName(value?: unknown) {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

export function getChiefComplaintInfo(node: PatientGraphNode): ChiefComplaintInfo {
  const keys = [normalizeComplaintName(node.original_feature_name), normalizeComplaintName(node.label)];
  for (const key of keys) {
    if (key && complaintInfo[key]) {
      return complaintInfo[key];
    }
  }

  return {
    displayName: 'Chief complaint',
    description: 'The main symptom or concern that brought the patient to emergency care.',
  };
}
