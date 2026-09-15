"""Evidence-driver guards and real sklearn train-only fitting regression."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'comparison/standardized/performance_review_20260913/evidence.py'

def driver():
    spec = importlib.util.spec_from_file_location('review_evidence', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_baseline_fits_training_only_and_refuses_occupied_output(tmp_path, monkeypatch):
    import sklearn.linear_model
    from sklearn.linear_model import LogisticRegression
    module = driver(); module.ROOT = tmp_path
    a=tmp_path/'audit'; a.mkdir()
    x=np.tile(np.eye(30,dtype=np.float32),(3,1))
    y=np.tile(np.arange(30),3); folds=np.repeat(np.arange(3),30)
    x[30:60]*=2; x[60:]*=100
    np.savez(a/'features_no_identifiers.npz',x=x,y=y,folds=folds)
    (a/'coverage.json').write_text(json.dumps({'concept_dim':30}))
    fits=[]; predictions=[]
    original_fit=LogisticRegression.fit
    original_predict=LogisticRegression.predict_proba
    def traced_fit(self, X, y, *args, **kwargs):
        fits.append(np.array(X)); return original_fit(self,X,y,*args,**kwargs)
    def traced_predict(self,X):
        predictions.append(np.array(X)); return original_predict(self,X)
    monkeypatch.setattr(LogisticRegression,'fit',traced_fit)
    monkeypatch.setattr(LogisticRegression,'predict_proba',traced_predict)
    module.baseline()
    assert len(fits)==4 and all(np.array_equal(z,x[:30]) for z in fits)
    assert all(np.array_equal(z,x[30:60]) for z in predictions)
    state=json.loads((tmp_path/'baseline/state.json').read_text())
    assert state['status']=='completed' and state['test_evaluated'] is False
    with pytest.raises(FileExistsError): module.baseline()

def test_training_budget_rejected_before_any_artifact(tmp_path):
    module=driver(); module.ROOT=tmp_path
    with pytest.raises(ValueError): module.experiment('gsat','current',4,'train')
    assert list(tmp_path.iterdir())==[]

def test_atomic_json_save(tmp_path):
    module=driver(); path=tmp_path/'state.json'
    module.save(path,{'status':'running'}); module.save(path,{'status':'completed'})
    assert json.loads(path.read_text())=={'status':'completed'}
    assert not path.with_suffix('.json.tmp').exists()
