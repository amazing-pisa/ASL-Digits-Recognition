"""Portable commands for validating, evaluating and training the released artifacts."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/core'))
sys.path.insert(0,str(ROOT/'models'))
ARTIFACT_ROOT=Path(os.environ.get('ASL_ARTIFACT_ROOT',str(ROOT))).resolve()

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_data():
    import numpy as np
    rows=read(ROOT/'data/splits/split_manifest.csv')
    sources=read(ROOT/'data/splits/source_data_manifest.csv')
    arrays={r['file']:np.load(ARTIFACT_ROOT/'data/dataset'/r['file'],allow_pickle=False) for r in sources}
    x=np.asarray([arrays[r['file']][int(r['file_index']),:,:212] for r in rows],dtype=np.float32)
    y=np.asarray([int(r['label']) for r in rows])
    assert x.shape==(976,30,212) and np.isfinite(x).all()
    for r in rows:assert np.all(arrays[r['file']][int(r['file_index']),:,-1]==int(r['label']))
    return x,y,rows

def validate():
    import ast
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    for row in read(ROOT/'data/splits/source_data_manifest.csv'):
        assert sha(ARTIFACT_ROOT/'data/dataset'/row['file'])==row['sha256'],row['file']
    x,y,rows=load_data()
    assert len({r['sample_id'] for r in rows})==976
    assert {s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}==dict(train=682,validation=98,test=196)
    assert not list(ROOT.rglob('*.avi')) and not list(ROOT.rglob('*.mp4'))
    print('PASS: code syntax, 976 feature sequences, exact 682/98/196 memberships, data hashes; recorded routing/latency evaluation data are not distributed')

def evaluate(family):
    import numpy as np
    import tensorflow as tf
    from model_builders import build_model
    from sklearn.metrics import accuracy_score,f1_score
    x,y,rows=load_data();test=[i for i,r in enumerate(rows) if r['split']=='test'];truth=y[test]
    if family=='architectures':
        group='dynamic_architectures';models=['stcnn','glt','lite'];builder=build_model
    else:
        group='delta_capacity';models=['full_delta','no_delta_original','no_delta_matched']
        spec=importlib.util.spec_from_file_location('capacity',ROOT/'scripts/experiments/delta_capacity/run_experiment.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        builder=lambda name:helper.build_model(*helper.VARIANTS[name])
    results=[]
    for name in models:
        for seed in [2021,2022,2023]:
            tf.keras.backend.clear_session();model=builder(name)
            model.load_weights(ARTIFACT_ROOT/f'weights/{group}/{name}_seed{seed}/best.weights.h5')
            dim=int(model.input_shape[-1]);features=x[test,:,:dim]
            prob=np.concatenate([model(features[i:i+32],training=False).numpy() for i in range(0,len(test),32)])
            pred=prob.argmax(axis=1)
            saved=read(ROOT/f'results/{group}/{name}_seed{seed}/predictions.csv')
            expected={r['sample_id']:int(r['predicted_digit']) for r in saved}
            assert all(int(p)+10==expected[rows[i]['sample_id']] for p,i in zip(pred,test)),(name,seed,'Predictions differ from archived results')
            record=dict(model=name,seed=seed,accuracy=float(accuracy_score(truth,pred)),macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)))
            result=json.loads((ROOT/f'results/{group}/{name}_seed{seed}/result.json').read_text(encoding='utf-8'))
            assert abs(record['accuracy']-result['accuracy'])<1e-12 and abs(record['macro_f1']-result['macro_f1'])<1e-12
            results.append(record);print(record,flush=True)
    dest=ROOT/'reproduced';dest.mkdir(exist_ok=True)
    (dest/f'{group}_evaluation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

def analyze_motion():
    raise FileNotFoundError('Recorded motion-session data are not distributed. Use scripts/experiments/motion_validation/threshold_sensitivity.py --session YOUR_SESSION with your own records.')

def analyze_latency():
    raise FileNotFoundError('Recorded latency-session data are not distributed. Use scripts/experiments/latency_validation/latency_test.py analyze --session YOUR_SESSION with your own records.')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','evaluate','train','motion','latency','camera']);p.add_argument('--family',choices=['architectures','ablation'],default='architectures');args=p.parse_args()
    if args.command=='validate':validate()
    elif args.command=='evaluate':evaluate(args.family)
    elif args.command=='motion':analyze_motion()
    elif args.command=='latency':analyze_latency()
    elif args.command=='camera':subprocess.run([sys.executable,str(ROOT/'scripts/core/camera_app.py')],check=True)
    else:
        group='dynamic_architectures' if args.family=='architectures' else 'delta_capacity'
        subprocess.run([sys.executable,str(ROOT/f'scripts/experiments/{group}/run_experiment.py')],check=True)

if __name__=='__main__':main()
