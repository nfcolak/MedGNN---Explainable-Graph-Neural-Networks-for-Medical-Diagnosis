"""One-time audited compatibility migration; no model/input changes allowed."""
import hashlib
import json
from pathlib import Path
from comparison.standardized.common_input_improvement import ROOT
from comparison.standardized.performance_review_20260913.evidence import save

def sha(b): return hashlib.sha256(b).hexdigest()
p=Path('comparison/standardized/common_input_improvement.py')
adapter=p.read_text()
old_adapter=adapter.replace("hasattr(torch, 'mps') and torch.backends.mps.is_available()", "torch.backends.mps.is_available()")
start=old_adapter.index('def load_checkpoint(path):')
end=old_adapter.index('def restore_checkpoint(path, model, optimizer):')
old_adapter=old_adapter[:start]+old_adapter[end:]
old_adapter=old_adapter.replace('    state = load_checkpoint(path)', "    state = torch.load(path, map_location='cpu', weights_only=False)")
p=Path('comparison/standardized/run_common_input.py'); runner=p.read_text()
old_runner=runner.replace('training_phase, load_checkpoint)', 'training_phase)')
old_runner=old_runner.replace("ck=load_checkpoint(target/'last.pt')", "ck=torch.load(target/'last.pt',map_location='cpu',weights_only=False)")
s=ROOT/'trials/gsat/current/state.json'; state=json.loads(s.read_text())
assert state['status']=='interrupted' and state['next_epoch']==1
assert state['binding']['adapter_sha256']==sha(old_adapter.encode())
assert state['binding']['harness_sha256']==sha(old_runner.encode())
save(s.parent/'state_before_compatibility_migration.json',state)
last=s.parent/'last.pt'; old_ck_hash=sha(last.read_bytes())
old_binding=dict(state['binding'])
state['binding']['adapter_sha256']=sha(adapter.encode())
state['binding']['harness_sha256']=sha(runner.encode())
state['compatibility_migration']='../../../../compatibility_migration.json'
save(s,state)
assert sha(last.read_bytes())==old_ck_hash
failed=ROOT/'trials/graphcare/current'
assert failed.exists() and not (failed/'last.pt').exists()
archive=ROOT/'failed_attempts/graphcare_current_precheckpoint'; archive.parent.mkdir(exist_ok=True)
failed.rename(archive)
state_gc=json.loads((archive/'state.json').read_text())
save(archive/'state_before_reconciliation.json',state_gc)
state_gc['status']='failed'; state_gc['error']='Legacy Torch has no torch.mps RNG API; failed before initial checkpoint or first update.'
save(archive/'state.json',state_gc)
(ROOT/'graphcare_current.log').rename(ROOT/'graphcare_current_failed_precheckpoint.log')
save(ROOT/'compatibility_migration.json',{'reason':'Checkpoint API capability detection only. No input/model/loss/training changes.',
 'verified_old_sources_reconstructed':True,'old_binding':old_binding,'new_binding':state['binding'],
 'gsat_last_checkpoint_unchanged_sha256':old_ck_hash,'gsat_resume_from_epoch':1,'gsat_updates':466,
 'graphcare_failed_before_checkpoint':True,'graphcare_archive':str(archive),
 'graphcare_next_action':'explicit restart from seed1234, zero updates preserved; second chain cell never launched',
 'regression':'checkpoint restore failed on legacy Torch first, then passed in both interpreters'})
print('Compatibility migration verified; GSAT epoch1 preserved; GraphCare pretraining failure archived.')
