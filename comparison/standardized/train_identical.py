"""Single current comparison launcher: all methods on exact native ProtGNN/GSAT input.

Dry-run instantiates each full-data production runner in its actual interpreter;
no output directory is created. --limit is only an explicitly labelled wiring proof.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from comparison.standardized.native_reference_v1.data import Reference, DEFAULT, REPO, save, digest
from comparison.standardized.native_reference_v1.models import METHODS


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--artifact',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--loss',choices=['native','ce','sqrt_inverse'],default='ce')
    p.add_argument('--limit',type=int);p.add_argument('--stop-after',type=int)
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--replay',action='store_true')
    a=p.parse_args(argv)
    if a.execute and a.dry_run:p.error('Choose execute or dry-run')
    if len(a.methods)!=len(set(a.methods)):p.error('Duplicate methods')
    ref=Reference(a.artifact)
    for method in a.methods:
        python=str(REPO/'.venv-graphcare/bin/python') if method=='graphcare' else shutil.which('python3')
        command=[python,'-m','comparison.standardized.native_reference_v1.run','--method',method,
            '--artifact',str(a.artifact.resolve()),'--expected-contract',ref.fingerprint,'--output',str(a.output.resolve()/method),
            '--epochs',str(a.epochs),'--seed',str(a.seed),'--batch-size',str(a.batch_size),'--loss',a.loss]
        for key in ['limit','stop_after']:
            if getattr(a,key) is not None:command.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
        command.append('--execute' if a.execute else '--dry-run')
        for key in ['resume','replay']:
            if getattr(a,key):command.append('--'+key)
        subprocess.run(command,cwd=REPO,check=True)
    if a.execute:
        states=[json.loads((a.output/m/'run_manifest.json').read_text()) for m in a.methods]
        cohort_keys=['contract_sha256','seed','batch_size','epochs','limit','loss','train_ordinals_sha256','validation_ordinals_sha256']
        for k in cohort_keys:
            if len({json.dumps(s['binding'][k],sort_keys=True) for s in states})!=1:raise ValueError('Cross-method '+k+' differs')
        histories=[json.loads((a.output/m/'history.json').read_text()) for m in a.methods]
        if len({digest([r['order_sha256'] for r in history]) for history in histories})!=1:raise ValueError('Actual training order differs')
        result={'methods':a.methods,'states':{m:s['status'] for m,s in zip(a.methods,states)},'input_sha256':ref.contract['input_sha256'],
                'contract_sha256':ref.fingerprint,'identical_cohorts_and_training_order':True,'counts':ref.contract['counts'],
                'selected_counts':states[0]['selected_counts'],'scope':states[0]['scope'],'test_evaluated':False}
        save(a.output/'integration.json',result);print(json.dumps(result))


if __name__=='__main__':main()
