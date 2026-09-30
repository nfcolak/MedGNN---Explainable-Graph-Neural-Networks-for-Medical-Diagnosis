"""Clinical decision-point graph, version 2.

Rebuilt from scratch after the v1 event-graph diagnosis (docs/new-input-diagnosis.md)
showed that every v1 edge was either membership bookkeeping (`contains_event`,
`observes_concept`) or a time ordering of nodes that already carry their own
timestamps. Measured consequence: the whole v1 topology added +0.009 macro-F1 over
its own node features, and `structure_only` scored below the majority baseline.

v2 changes three things and nothing else:

1. Restores the decision-time evidence v1 discarded -- chief complaint, triage
   vitals and arrival demographics -- under the SAME strict cutoff rule.
2. Replaces bookkeeping edges with relations whose endpoints must be compared for
   the edge to mean anything (same-analyte baselines carrying deltas, complaint
   conjunctions). A fixed-width feature row cannot express these.
3. Drops the historical event dump that produced 18 GB for no measured signal.

Cohort, folds, cutoffs and sample ids are inherited byte-identical from the v1
artifact so that any comparison isolates the graph contract. This artifact is a
separate input version; it never claims parity with the legacy benchmark.
"""

SCHEMA_VERSION = 'clinical_graph_v2'

NODE_KINDS = ('patient', 'visit', 'complaint', 'measurement', 'analyte', 'vital',
              'knowledge', 'diagnosis')

# Relations are split into two classes on purpose. Only `INFORMATIVE` relations may
# be cited as a reason for using a graph model; `STRUCTURAL` relations exist so the
# graph is navigable and are expected to be reconstructible from node features.
STRUCTURAL_RELATIONS = ('has_visit', 'index_visit_of', 'reports_complaint',
                        'measured_in', 'instance_of', 'observed_vital',
                        'has_prior_diagnosis')
INFORMATIVE_RELATIONS = ('baseline_of', 'trajectory_of', 'co_complaint',
                         'recurrence_of', 'comorbid_with',
                         'medical:member_of', 'medical:assesses', 'medical:measures')
