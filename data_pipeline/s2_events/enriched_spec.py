"""Closed allowlist transcribed from audited merge_ed/extract_ed_labs sources.

No prefix discovery of model features. Source units for labs were discarded by
legacy extraction; do not claim unit harmonization or result-time eligibility.
"""
LAB_ITEMS = {
    'albumin': 50862, 'alk_phos': 50863, 'alt': 50861, 'anion_gap': 50868,
    'ast': 50878, 'bicarbonate': 50882, 'bilirubin': 50885, 'bun': 51006,
    'calcium': 50893, 'chloride': 50902, 'creatinine': 50912, 'glucose': 50931,
    'hematocrit': 51221, 'hemoglobin': 51222, 'inr': 51237, 'lactate': 50813,
    'magnesium': 50960, 'mcv': 51250, 'phosphate': 50970, 'platelet': 51265,
    'potassium': 50971, 'ptt': 51275, 'rbc': 51279, 'sodium': 50983,
    'troponin_t': 51003, 'wbc': 51301,
}
HISTORY = [
    'hypertensive_urgency', 'type_2_diabetes_mellitus_without_complications',
    'low_back_pain', 'familial_hypercholesterolemia', 'pain_in_unspecified_limb',
    'pain_in_unspecified_knee', 'essential_primary_hypertension', 'urinary_tract_infection',
    'pneumonia', 'cellulitis_of_unspecified_part_of_limb', 'alcohol_abuse_with_intoxication',
    'hypokalemia', 'major_depressive_disorder', 'acute_kidney_failure',
    'athscl_heart_disease_of_native_coronary_artery_w_o_ang_pctrs',
    'other_specified_injuries_of_head', 'type_1_diabetes_mellitus_without_complications',
    'gastrointestinal_hemorrhage', 'heart_failure', 'dehydration',
    'unspecified_atrial_fibrillation', 'anemia',
    'chronic_obstructive_pulmonary_disease_w_acute_exacerbation',
    'unspecified_asthma_with_acute_exacerbation', 'end_stage_renal_disease',
    'other_chronic_pain', 'anxiety_disorder', 'contusion_of_unspecified_part_of_head',
    'unspecified_open_wound_of_other_part_of_head', 'acute_upper_respiratory_infection',
]
VITALS = {'temperature': ('Celsius', [25., 45.]), 'heartrate': ('beats/min', [5., 300.]),
          'o2sat': ('percent', [50., 100.]), 'sbp': ('mmHg', [40., 300.]),
          'dbp': ('mmHg', [0., 200.])}
SPEC = [{'source': 'age', 'kind': 'numeric', 'unit': 'years (anchor_age, not index age)',
         'availability': 'anchor_demographic_snapshot'}]
SPEC += [{'source': n, 'kind': 'numeric', 'unit': unit, 'range': bounds,
          'availability': 'initial_triage_measurement_no_recording_timestamp'}
         for n, (unit, bounds) in VITALS.items()]
SPEC += [{'source': 'lab_' + n, 'kind': 'numeric', 'itemid': item,
          'unit': 'source numeric unit unverified; valueuom discarded upstream',
          'availability': 'whole_stay_mean_result_availability_unknown',
          'early_triage_eligible': False} for n, item in LAB_ITEMS.items()]
SPEC += [{'source': 'hx_' + n, 'kind': 'binary', 'unit': 'indicator',
          'availability': 'historical_source_summary_documentation_time_unverified',
          'early_triage_eligible': False} for n in HISTORY]
ALLOWED = {f['source']: f['kind'] for f in SPEC}
