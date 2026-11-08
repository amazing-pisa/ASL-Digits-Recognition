"""Exploratory causal-score classifiers with nested, trial-grouped CV.

Only past/current movement scores are inputs. No digit, trial identity, video
phase, future frames, predictions or labels are features. Original logs remain
read-only. Existing session has been inspected previously: not a new blind test.
"""
from pathlib import Path
from collections import deque
import csv
import hashlib
import json
import argparse
import math
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parent
SEED=20261107
WINDOWS=[3,5,10,20,30]
PROB_THRESHOLDS=[.35,.5,.65]
KINDS=['logistic','hist7','hist15','forest']

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def features(frames):
    history=deque(maxlen=30)
    previous_time=None
    result=[]
    for r in frames:
        if r['hand_detected']!='1' or r['route'] not in ('static','dynamic') or not r['motion_score']:
            history.clear();previous_time=None;result.append(None);continue
        now=float(r['timestamp'])
        # Reset causal context on detection gaps; never use future measurements.
        if previous_time is not None and now-previous_time>.5:history.clear()
        previous_time=now
        score=float(r['motion_score']);assert score>=0 and math.isfinite(score)
        value=math.log10(max(score,1e-10));history.append(value)
        a=np.asarray(history,dtype=np.float64)
        row=[value]
        for n in WINDOWS:
            w=a[-n:]
            row.extend([float(w.mean()),float(w.std()),float(w.min()),float(w.max()),float(w[-1]-w[0])])
        result.append(row)
    return result

def build(kind):
    if kind=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=SEED))
    if kind in ('hist7','hist15'):
        return HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=7 if kind=='hist7' else 15,
                                              min_samples_leaf=30,l2_regularization=1.,early_stopping=False,random_state=SEED)
    return RandomForestClassifier(n_estimators=100,max_depth=7,min_samples_leaf=20,max_features='sqrt',n_jobs=1,random_state=SEED)

def metrics(y,p):
    ss,sd,ds,dd=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    return dict(frames=len(y),true_static_pred_static=ss,true_static_pred_dynamic=sd,true_dynamic_pred_static=ds,true_dynamic_pred_dynamic=dd,
                accuracy_percent=100*(ss+dd)/len(y),balanced_accuracy_percent=50*(ss/(ss+sd)+dd/(ds+dd)),
                static_false_dynamic_percent=100*sd/(ss+sd),dynamic_false_static_percent=100*ds/(ds+dd))

def run(session,out):
    sources={n:sha(session/n) for n in ['frames.csv','annotations.csv']}
    frames,annotations=read(session/'frames.csv'),read(session/'annotations.csv')
    assert len(frames)==len(annotations) and all(f['frame']==a['frame'] for f,a in zip(frames,annotations))
    full_x=features(frames)
    ids=[i for i,r in enumerate(frames) if r['phase']=='active' and full_x[i] is not None and annotations[i]['motion_truth'] in ('static','dynamic')]
    x=np.asarray([full_x[i] for i in ids],dtype=np.float64)
    y=np.asarray([int(annotations[i]['motion_truth']=='dynamic') for i in ids])
    groups=np.asarray([int(frames[i]['trial']) for i in ids])
    assert x.shape==(len(ids),26) and np.isfinite(x).all()
    original=np.asarray([int(frames[i]['route']=='dynamic') for i in ids])
    assert all(original[k]==int(float(frames[i]['motion_score'])>=.0005) for k,i in enumerate(ids))
    old=json.loads((session/'routing_report.json').read_text(encoding='utf-8'))
    assert len(ids)==old['scored_frames']
    assert abs(metrics(y,original)['accuracy_percent']/100-old['conditional_accuracy'])<1e-12
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(session=session.name,input_sha256=sources,script_sha256=sha(Path(__file__)),seed=SEED,
                  feature_scope='26 causal features of log10 current/past motion scores only; history reset on invalid detection or gaps over .5 seconds',
                  windows=WINDOWS,candidates=KINDS,probability_thresholds=PROB_THRESHOLDS,
                  design='5 outer folds and 3 inner folds; StratifiedGroupKFold by complete trial',
                  objective='inner out-of-fold accuracy, then balanced accuracy; no outer test scores used for selection',
                  eligible='identical valid annotated active frames across all methods',
                  scope='exploratory post-hoc reanalysis of previously inspected single-session data; not a new blind recording',
                  exclusions='no future scores, gesture identity, trial ID, scripted condition, absolute time, phase, model confidence or output digit as features',
                  deployment='new supervised routing method, not a remeasurement of original threshold mechanism; not connected to live system',
                  package_versions=dict(numpy=np.__version__,sklearn=__import__('sklearn').__version__))
    save(out/'protocol.json',protocol)
    prediction=np.full(len(y),-1,dtype=int);probability=np.full(len(y),np.nan);fold_ids=np.full(len(y),-1,dtype=int)
    search,fold_rows,splits=[],[],[]
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    for fold,(train,test) in enumerate(outer.split(x,y,groups),1):
        assert not set(groups[train]) & set(groups[test])
        inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+fold).split(x[train],y[train],groups[train]))
        best=None
        for kind in KINDS:
            inner_p=np.full(len(train),np.nan)
            for tr,va in inner:
                assert not set(groups[train[tr]]) & set(groups[train[va]])
                model=build(kind);model.fit(x[train[tr]],y[train[tr]])
                inner_p[va]=model.predict_proba(x[train[va]])[:,1]
            assert np.isfinite(inner_p).all()
            for threshold in PROB_THRESHOLDS:
                m=metrics(y[train],(inner_p>=threshold).astype(int))
                search.append(dict(outer_fold=fold,kind=kind,threshold=threshold,**m))
                key=(m['accuracy_percent'],m['balanced_accuracy_percent'])
                if best is None or key>best[0]:best=(key,kind,threshold)
        _,kind,threshold=best
        # Persist the selected setting before evaluating this outer fold.
        splits.append(dict(fold=fold,training_trials=sorted(map(int,set(groups[train]))),test_trials=sorted(map(int,set(groups[test]))),kind=kind,threshold=threshold))
        save(out/'fold_choices.json',splits)
        model=build(kind);model.fit(x[train],y[train]);p=model.predict_proba(x[test])[:,1]
        prediction[test]=(p>=threshold).astype(int);probability[test]=p;fold_ids[test]=fold
        joblib.dump(model,out/f'fold{fold}_model.joblib')
        for name,pred in [('original',original[test]),('learned',prediction[test]),('always_static',np.zeros(len(test),dtype=int))]:
            fold_rows.append(dict(fold=fold,method=name,**metrics(y[test],pred)))
        print('FOLD',fold,'selected',kind,threshold,'accuracy',metrics(y[test],prediction[test])['accuracy_percent'],flush=True)
    assert (prediction>=0).all() and (fold_ids>0).all() and np.isfinite(probability).all()
    for group in set(groups):assert len(set(fold_ids[groups==group]))==1
    summary={name:metrics(y,p) for name,p in [('original',original),('learned',prediction),('always_static',np.zeros(len(y),dtype=int))]}
    csvout(out/'summary.csv',[dict(method=k,**v) for k,v in summary.items()])
    csvout(out/'fold_metrics.csv',fold_rows);csvout(out/'inner_search.csv',search)
    csvout(out/'predictions.csv',[dict(frame=frames[i]['frame'],trial=int(groups[k]),fold=int(fold_ids[k]),truth=int(y[k]),original=int(original[k]),learned=int(prediction[k]),prob_dynamic=float(probability[k])) for k,i in enumerate(ids)])
    trials=[]
    for t in sorted(set(groups)):
        m=groups==t
        trials.append(dict(trial=int(t),fold=int(fold_ids[m][0]),frames=int(m.sum()),original_accuracy_percent=100*float(np.mean(original[m]==y[m])),learned_accuracy_percent=100*float(np.mean(prediction[m]==y[m]))))
    csvout(out/'per_trial.csv',trials)
    save(out/'results.json',dict(protocol=protocol,summary=summary,
                               trials_improved=sum(r['learned_accuracy_percent']>r['original_accuracy_percent'] for r in trials),
                               trials_worse=sum(r['learned_accuracy_percent']<r['original_accuracy_percent'] for r in trials)))
    assert sources=={n:sha(session/n) for n in sources}
    lines=['# 学习型运动路由探索结果','',
           '在相同原始标签和运动分数上比较阈值路由与新的监督分类器。共42个完整试次，使用五折外层、三折内层分组交叉验证；每帧仅由未使用其所在试次训练的模型预测。内层选择模型和概率阈值，外层仅评价，未使用未来运动分数或手势类别作为输入。','',
           '|方法|准确率（%）|平衡准确率（%）|静态误判为动态（%）|动态误判为静态（%）|',
           '|---|---:|---:|---:|---:|']
    names={'original':'原阈值路由','learned':'学习型路由（逐帧折外预测）','always_static':'全部判静态（参照）'}
    for name,m in summary.items():lines.append('|'+names[name]+'|'+'|'.join(f'{m[k]:.2f}' for k in ['accuracy_percent','balanced_accuracy_percent','static_false_dynamic_percent','dynamic_false_static_percent'])+'|')
    lines+=['',f"新路由相对原阈值的总体准确率变化为{summary['learned']['accuracy_percent']-summary['original']['accuracy_percent']:+.2f}个百分点。",'',
            '全部判静态作为参照，是为了检查总体准确率提高是否仅来自多数类偏向；需要同时查看两类误判。',
            '这是同一已查看过会话上的探索性方法比较，不能称为全新独立盲测，也没有证明多参与者泛化或统计显著性。新路由属于不同算法，不能把它的混淆矩阵用于声称原阈值算法取得该效果。未更改原标签、实时系统、论文或发布包；未重新评估最终数字识别和端到端延迟。','',
            '每个外层模型见fold*_model.joblib；分组与所选参数见fold_choices.json；逐帧折外预测见predictions.csv；逐试次表现见per_trial.csv。']
    (out/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:12]),flush=True)
    purposes={'protocol.json':'数据校验、因果特征、分组设计、候选模型和选择规则',
              'fold_choices.json':'每折训练／测试试次及内层选定参数',
              'summary.csv':'相同样本上的阈值、学习型和全静态基线指标',
              'fold_metrics.csv':'每个外层测试折的比较指标',
              'inner_search.csv':'仅在内层折外预测上计算的参数选择记录',
              'predictions.csv':'每帧真值及阈值、折外分类预测和概率',
              'per_trial.csv':'每个完整试次的两种算法准确率',
              'results.json':'完整结果及改善／退步的试次数',
              '结果说明.md':'评价方法、结果和使用范围'}
    files=[Path(__file__).resolve()]+sorted(out.iterdir())
    manifest=[]
    for p in files:
        purpose='监督运动分类器的因果特征和嵌套分组交叉验证入口' if p.suffix=='.py' else '对应外层折模型：由训练试次拟合，用于该折留出预测' if p.suffix=='.joblib' else purposes[p.name]
        manifest.append(dict(file=str(p.resolve()),purpose=purpose,sha256=sha(p)))
    manifest.append(dict(file=str((out/'文件用途清单.csv').resolve()),purpose='本次每个文件的用途和校验值',sha256='self'))
    csvout(out/'文件用途清单.csv',manifest)
    print('INPUTS_UNCHANGED; SAVED',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',type=Path,default=ROOT.parents[2]/'data/motion')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    args=parser.parse_args();run(args.session,args.output)
