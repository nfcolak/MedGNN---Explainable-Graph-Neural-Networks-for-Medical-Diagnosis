"""One-off authorized cache-only execution evidence; never launches training."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'comparison/standardized' / ('production_cache_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
OUT.mkdir(exist_ok=False)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

inputs = [ROOT/'data/merged_ed.csv', ROOT/'comparison/canonical_split.json']
legacy = list((ROOT/'data').rglob('*.pt'))
protected = inputs + legacy
before = {str(p.relative_to(ROOT)): {'sha256': digest(p), 'symlink': p.is_symlink(), 'target': str(p.resolve()), 'bytes': p.stat().st_size} for p in protected}
state = {'scope': 'Production caches and all-canonical-subject parity only; no training/explanations', 'started': datetime.datetime.now().astimezone().isoformat(), 'python': sys.executable, 'protected_before': before, 'commands': []}
(OUT/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
(OUT/'resources_before.txt').write_text(subprocess.check_output(['df','-h','.'],cwd=ROOT,text=True)+subprocess.check_output(['vm_stat'],text=True)+subprocess.check_output(['sysctl','hw.memsize','vm.swapusage'],text=True))

def checkpoint():
    temp=OUT/'checkpoint.tmp'
    temp.write_text(json.dumps(state,indent=2))
    temp.replace(OUT/'checkpoint.json')

checkpoint()
print('EVIDENCE_DIRECTORY',OUT,flush=True)
for name, argv in [
    ('build', [sys.executable,'-u','-m','comparison.standardized.build_caches','--execute','--structures','star','cooccur']),
    ('audit', [sys.executable,'-u','-m','comparison.standardized.audit_caches','--structures','star','cooccur']),
]:
    entry={'name':name,'argv':argv,'status':'running','log':str(OUT/(name+'.log'))}
    state['commands'].append(entry)
    checkpoint()
    start=time.monotonic()
    with (OUT/(name+'.log')).open('x') as log:
        result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    entry.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,elapsed_seconds=time.monotonic()-start)
    checkpoint()
    print(name,entry['status'],'elapsed_seconds',entry['elapsed_seconds'],flush=True)
    if result.returncode:
        break
state['protected_after']={str(p.relative_to(ROOT)): {'sha256':digest(p),'symlink':p.is_symlink(),'target':str(p.resolve()),'bytes':p.stat().st_size} for p in protected}
state['protected_unchanged']=state['protected_after']==before
state['finished']=datetime.datetime.now().astimezone().isoformat()
checkpoint()
print('PROTECTED_UNCHANGED',state['protected_unchanged'],flush=True)
