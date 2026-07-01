import type { PatientGraphNode } from '../types/graph';

export interface VitalInfo {
  displayName: string;
  description: string;
}

const vitalInfo: Record<string, VitalInfo> = {
  temperature: {
    displayName: 'Body temperature',
    description: 'Body temperature reflects the balance between heat production and heat loss. Fever can suggest infection or inflammation; low temperature can occur with exposure, shock, or severe illness.',
  },
  heartrate: {
    displayName: 'Heart rate',
    description: 'Heart rate is the number of heart beats per minute. It rises with pain, fever, dehydration, blood loss, anxiety, infection, or abnormal heart rhythms.',
  },
  resprate: {
    displayName: 'Respiratory rate',
    description: 'Respiratory rate is the number of breaths per minute. It can increase when the body needs more oxygen or must remove more carbon dioxide, such as in lung disease, sepsis, pain, or metabolic stress.',
  },
  o2sat: {
    displayName: 'Oxygen saturation',
    description: 'Oxygen saturation estimates the percentage of hemoglobin in the blood carrying oxygen. Low values can indicate impaired oxygen transfer, lung disease, airway problems, or poor perfusion.',
  },
  sbp: {
    displayName: 'Systolic blood pressure',
    description: 'Systolic blood pressure is the peak pressure in the arteries when the heart contracts. It helps assess circulation, shock, bleeding, dehydration, and hypertensive stress.',
  },
  dbp: {
    displayName: 'Diastolic blood pressure',
    description: 'Diastolic blood pressure is the pressure in the arteries while the heart relaxes between beats. It reflects vascular tone and contributes to overall blood pressure assessment.',
  },
  pain: {
    displayName: 'Pain score',
    description: 'Pain score is the patient’s reported intensity of pain. It helps contextualize distress, injury severity, analgesic need, and changes in vital signs such as heart rate or blood pressure.',
  },
  acuity: {
    displayName: 'Triage acuity',
    description: 'Triage acuity expresses how urgently the patient needs evaluation. It combines the presenting problem, risk, symptoms, and vital-sign concern into a severity level.',
  },
  vs_temperature: {
    displayName: 'Body temperature',
    description: 'Body temperature reflects the balance between heat production and heat loss. Fever can suggest infection or inflammation; low temperature can occur with exposure, shock, or severe illness.',
  },
  vs_heartrate: {
    displayName: 'Heart rate',
    description: 'Heart rate is the number of heart beats per minute. It rises with pain, fever, dehydration, blood loss, anxiety, infection, or abnormal heart rhythms.',
  },
  vs_resprate: {
    displayName: 'Respiratory rate',
    description: 'Respiratory rate is the number of breaths per minute. It can increase when the body needs more oxygen or must remove more carbon dioxide, such as in lung disease, sepsis, pain, or metabolic stress.',
  },
  vs_o2sat: {
    displayName: 'Oxygen saturation',
    description: 'Oxygen saturation estimates the percentage of hemoglobin in the blood carrying oxygen. Low values can indicate impaired oxygen transfer, lung disease, airway problems, or poor perfusion.',
  },
  vs_sbp: {
    displayName: 'Systolic blood pressure',
    description: 'Systolic blood pressure is the peak pressure in the arteries when the heart contracts. It helps assess circulation, shock, bleeding, dehydration, and hypertensive stress.',
  },
  vs_dbp: {
    displayName: 'Diastolic blood pressure',
    description: 'Diastolic blood pressure is the pressure in the arteries while the heart relaxes between beats. It reflects vascular tone and contributes to overall blood pressure assessment.',
  },
  vs_count: {
    displayName: 'Vital-sign count',
    description: 'Vital-sign count indicates how often the patient’s basic physiologic status was checked. More frequent checks often reflect closer monitoring or a changing clinical condition.',
  },
};

function normalizeVitalName(value?: unknown) {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

export function getVitalInfo(node: PatientGraphNode): VitalInfo {
  const keys = [normalizeVitalName(node.original_feature_name), normalizeVitalName(node.label)];
  for (const key of keys) {
    if (key && vitalInfo[key]) {
      return vitalInfo[key];
    }
  }

  return {
    displayName: 'Vital sign',
    description: 'A vital sign is a basic physiologic measurement used to assess the patient’s current clinical state.',
  };
}
