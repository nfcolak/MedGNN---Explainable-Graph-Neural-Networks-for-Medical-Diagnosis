export type PatientNodeType =
  | 'patient'
  | 'vital'
  | 'medication'
  | 'med'
  | 'chief_complaint'
  | 'cc'
  | 'diagnosis_attribute'
  | 'symptom'
  | 'icd';

export type PatientEdgeType = 'patient_feature' | 'pmi_bundle' | 'star' | 'pmi';

export interface PatientGraphNode {
  id: string | number;
  label: string;
  type: PatientNodeType;
  value?: number | string | null;
  z?: number | null;
  original_feature_name?: string;
  importance?: number;
  abnormal?: boolean;
  unit?: string;
  raw_value?: number | string;
  description?: string;
  [key: string]: unknown;
}

export interface PatientGraphEdge {
  source?: string | number;
  target?: string | number;
  a?: string | number;
  b?: string | number;
  type?: PatientEdgeType;
  kind?: PatientEdgeType;
  pmi?: number;
  weight?: number;
  [key: string]: unknown;
}

export interface PatientGraph {
  graph_id?: string | number;
  index?: string | number;
  dataset_index?: string | number;
  subject_id?: string | number;
  diagnosis: string;
  nodes: PatientGraphNode[];
  edges: PatientGraphEdge[];
}

export interface GraphManifestItem {
  id: string;
  label: string;
  file: string;
  graph_id?: string | number;
  diagnosis?: string;
  subject_id?: string | number;
  node_count?: number;
  edge_count?: number;
}

export interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  node?: PatientGraphNode;
}
