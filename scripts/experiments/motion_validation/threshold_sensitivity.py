"""Replay fixed, recorded scores at preset thresholds; no fitting or relabelling."""
from pathlib import Path
import argparse,csv,hashlib,json

ROOT=Path(__file__).resolve().parent
THRESHOLDS=[.0001,.00025,.0005,.001,.002]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main(session):
    frames,labels=read(session/'frames.csv'),read(session/'annotations.csv')
    assert len(frames)==len(labels)
    data=[]
    for r,a in zip(frames,labels):
        assert r['frame']==a['frame']
        if r['phase']=='active' and r['hand_detected']=='1' and a['motion_truth'] in ['static','dynamic'] and r['route'] in ['static','dynamic']:
            score=float(r['motion_score'])
            assert ('static' if score<.0005 else 'dynamic')==r['route']
            data.append((a['motion_truth'],score))
    assert data
    results=[]
    for threshold in THRESHOLDS:
        cm={t:{p:0 for p in ['static','dynamic']} for t in ['static','dynamic']}
        for truth,score in data:cm[truth]['static' if score<threshold else 'dynamic']+=1
        ns=sum(cm['static'].values());nd=sum(cm['dynamic'].values())
        assert ns and nd and ns+nd==len(data)
        results.append(dict(threshold=threshold,valid_frames=len(data),true_static_pred_static=cm['static']['static'],true_static_pred_dynamic=cm['static']['dynamic'],true_dynamic_pred_static=cm['dynamic']['static'],true_dynamic_pred_dynamic=cm['dynamic']['dynamic'],accuracy_percent=100*(cm['static']['static']+cm['dynamic']['dynamic'])/len(data),static_false_dynamic_percent=100*cm['static']['dynamic']/ns,dynamic_false_static_percent=100*cm['dynamic']['static']/nd))
    # The baseline must reproduce the previously reported raw confusion matrix.
    baseline=results[2]
    if (session/'routing_report.json').exists():
        previous=json.loads((session/'routing_report.json').read_text(encoding='utf-8'))
        cm=previous['confusion_matrix']
        for truth,pred,key in [('static','static','true_static_pred_static'),('static','dynamic','true_static_pred_dynamic'),('dynamic','static','true_dynamic_pred_static'),('dynamic','dynamic','true_dynamic_pred_dynamic')]:assert baseline[key]==cm[truth][pred]
    out=session/'threshold_sensitivity';out.mkdir(exist_ok=True)
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    report=dict(method='post-hoc threshold replay on identical recorded scores and manually labelled eligible frames',thresholds=THRESHOLDS,baseline_threshold=.0005,valid_frames=len(data),sources={n:sha(session/n) for n in ['frames.csv','annotations.csv']},script_sha256=sha(Path(__file__)),results=results,scope='instantaneous movement-state frame labels; temporally correlated frames, not independent gesture trials',selection='existing threshold retained; no new threshold selected on these evaluation labels',limits='No new live acquisition or full-system inference at alternative thresholds. Does not reconstruct historical threshold calibration or measure downstream recognition changes.')
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    print('BASELINE_REPRODUCED; INPUT_LABELS_AND_RECORDED_ROUTES_UNCHANGED',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,default=ROOT.parents[2]/'data/motion');a=p.parse_args();main(a.session)
