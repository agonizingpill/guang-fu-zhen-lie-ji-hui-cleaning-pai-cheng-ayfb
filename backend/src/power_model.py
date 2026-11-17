import hashlib,json,os,pickle
from functools import lru_cache
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
FEATURES=['capacity_kw','irradiance','module_temp','slope']
METHOD='RandomForestRegressor / 100 trees / max_depth 8'
ARTIFACTS=Path(os.getenv('ARTIFACTS_DIR','/artifacts'))/'models'
def train_candidate(rows):
    # random_state=42 的 80/20 划分为冻结留出集；留出样本不参与训练，供候选与现用模型同集比较。
    if len(rows)<40:raise ValueError('参考样本至少40条，才能形成可靠的冻结留出集')
    x=np.array([[float(r[k]) for k in FEATURES] for r in rows]);y=np.array([float(r['clean_kw']) for r in rows])
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.2,random_state=42)
    model=RandomForestRegressor(n_estimators=100,max_depth=8,min_samples_leaf=2,random_state=42,n_jobs=1)
    model.fit(train_x,train_y)
    metrics={'train_rows':len(train_x),'holdout_rows':len(test_x),'train_mae_kw':float(mean_absolute_error(train_y,model.predict(train_x))),'holdout_mae_kw':float(mean_absolute_error(test_y,model.predict(test_x))),'feature_importance':dict(zip(FEATURES,model.feature_importances_.tolist()))}
    return model,metrics,test_x,test_y
def holdout_mae(model,test_x,test_y):return float(mean_absolute_error(test_y,model.predict(test_x)))
def save_artifact(batch_id,model):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    tmp=ARTIFACTS/f'batch-{batch_id}.tmp';tmp.write_bytes(pickle.dumps(model));tmp.replace(ARTIFACTS/f'batch-{batch_id}.pkl')
@lru_cache(maxsize=8)
def _cached(batch_id,sha256,samples_blob):
    path=ARTIFACTS/f'batch-{batch_id}.pkl'
    if path.exists():
        try:return pickle.loads(path.read_bytes())
        except Exception:pass
    model,_,_,_=train_candidate(json.loads(samples_blob))
    try:save_artifact(batch_id,model)
    except OSError:pass
    return model
def load_model(batch):return _cached(batch.id,batch.sha256,json.dumps(batch.samples,sort_keys=True))
def predict(model,survey):
    clean=max(0,float(model.predict([[getattr(survey,k) for k in FEATURES]])[0]))
    return clean,max(0,clean-survey.observed_kw)
def evidence_for(batch,epoch):
    return {'batch_id':batch.id,'batch_code':batch.code,'epoch':epoch,'reference_rows':batch.sample_rows,'train_rows':batch.train_rows,'test_rows':batch.holdout_rows,'train_mae_kw':batch.train_mae_kw,'holdout_mae_kw':batch.holdout_mae_kw,'feature_importance':batch.feature_importance,'random_state':42,'sha256':batch.sha256,'method':METHOD}
def samples_sha(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
