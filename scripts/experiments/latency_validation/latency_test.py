"""Instrumented live inference; independent video-onset annotation; no training."""
from pathlib import Path
from collections import deque
from datetime import datetime
import sys, time, csv, json, uuid, tempfile, shutil, argparse, hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[2]/'scripts/core'))
import runtime
from camera_app import Recognizer, extract_features, cv2, mp, np
import tensorflow as tf

def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def tempdir():
    p=Path(tempfile.gettempdir())/('cmc_latency_'+uuid.uuid4().hex);p.mkdir();return p
def text(frame,lines):
    for i,s in enumerate(lines):cv2.putText(frame,s,(12,27+28*i),cv2.FONT_HERSHEY_SIMPLEX,.57,(0,255,0),2)
def keycode(k):
    k=k&255
    return k+32 if 65<=k<=90 else k
def latest():
    paths=sorted((ROOT/'sessions').glob('*/frames.csv'))
    if not paths:raise RuntimeError('Record a session first.')
    return paths[-1].parent

class TimedRecognizer(Recognizer):
    """Same decisions/state as repaired Recognizer; read clocks at stage boundaries."""
    def update(self,res):
        begin=time.perf_counter()
        timing=dict(feature_ms=0.,routing_ms=0.,sequence_features_ms=0.,inference_ms=0.,smoothing_ms=0.)
        result=dict(status='No hand',variance=None,candidate=None,stable=None,confidence=0.,inference_branch='')
        def finish():
            result.update(timing)
            result['recognizer_other_ms']=(time.perf_counter()-begin)*1000-sum(timing.values())
            result['sequence_size']=len(self.sequence)
            return result
        if res is None:self.reset();return finish()
        t=time.perf_counter();points,base=extract_features(res);timing['feature_ms']=(time.perf_counter()-t)*1000
        t=time.perf_counter();self.joints.append(points);self.sequence.append(base)
        if len(self.joints)<10:
            result['status']='Warming up';timing['routing_ms']=(time.perf_counter()-t)*1000;return finish()
        variance=float(np.var(np.asarray(self.joints),axis=0).mean())
        branch='Static' if self.dynamic is None or variance<float(runtime.CONFIG['motion_threshold']) else 'Dynamic'
        result.update(status=branch,variance=variance)
        if branch!=self.previous_branch:self.history.clear()
        self.previous_branch=branch
        timing['routing_ms']=(time.perf_counter()-t)*1000
        if branch=='Static':
            model=self.static;inputs=points[None,:,:];threshold=.8;offset=0
        else:
            if len(self.sequence)<30:
                result['status']='Accumulating dynamic frames';self.history.clear();return finish()
            t=time.perf_counter();bases=np.asarray(self.sequence,dtype=np.float32)
            delta=np.zeros_like(bases);delta[1:]=bases[1:]-bases[:-1]
            inputs=np.concatenate([bases,delta],axis=-1)[None,:,:]
            timing['sequence_features_ms']=(time.perf_counter()-t)*1000
            model=self.dynamic;threshold=.9;offset=10
        t=time.perf_counter();probability=model(inputs,training=False).numpy()[0]
        timing['inference_ms']=(time.perf_counter()-t)*1000
        result['inference_branch']=branch.lower()
        t=time.perf_counter();index=int(probability.argmax());confidence=float(probability[index]);candidate=str(index+offset)
        result.update(candidate=candidate,confidence=confidence)
        if confidence>threshold:self.history.append(candidate)
        else:self.history.clear()
        if len(self.history)==3 and len(set(self.history))==1:result['stable']=candidate
        timing['smoothing_ms']=(time.perf_counter()-t)*1000
        return finish()

def plan():
    return [dict(target=d,condition='reset_static',reset=True) for d in [0,1,2,3,4,5]]+[dict(target=d,condition='reset_dynamic',reset=True) for d in [10,11,12,15,18,20]]+[dict(target=d,condition='continuous_transition',reset=False) for d in [1,10,2,11,3,12,4,15,5,18,0,20]]

def record():
    session=ROOT/'sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');session.mkdir(parents=True)
    schedule=plan();trials=[];rows=[];annotations=[]
    model=TimedRecognizer()
    # Warm weights/runtime before timed acquisition, then reset only the temporal state.
    model.static(np.zeros((1,21,3),np.float32),training=False).numpy()
    model.dynamic(np.zeros((1,30,212),np.float32),training=False).numpy();model.reset()
    hashes={name:hashlib.sha256(((runtime.REPOSITORY/'models'/name) if name=='model_builders.py' else runtime.ROOT/name).read_bytes()).hexdigest() for name in ['camera_app.py','compat.py','model_builders.py','runtime.py']}
    hashes['configs/system.json']=hashlib.sha256((runtime.REPOSITORY/'configs/system.json').read_bytes()).hexdigest()
    hashes['latency_test.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    weights={k:dict(path=runtime.model_path(k),sha256=hashlib.sha256(Path(runtime.model_path(k)).read_bytes()).hexdigest()) for k in ['cnn','stcnn']}
    save(session/'protocol.json',dict(schedule=schedule,trial_seconds=8,threshold=runtime.CONFIG['motion_threshold'],warmup='one zero-input call each classifier; model load and first-call compilation excluded',onset='human-labelled first visible gesture movement in acquired video frame',endpoint='first correct stable result submitted to cv2.imshow after onset',physical_capture_and_display_latency_measured=False,source_sha256=hashes,weights=weights,python=sys.version,tensorflow=tf.__version__,keras=__import__('keras').__version__,mediapipe=mp.__version__,opencv=cv2.__version__,devices=[str(v) for v in tf.config.list_physical_devices()],video_container_fps=25))
    cap=cv2.VideoCapture(runtime.CONFIG.get('camera_index',0));hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.5,min_tracking_confidence=.5)
    tmp=tempdir();writer=None;idx=-1;active=False;clean=False
    window='Latency - record'
    try:
        if not cap.isOpened():raise RuntimeError('Cannot open camera. Close other camera applications.')
        while True:
            loop_start=time.perf_counter();t=loop_start;ok,raw=cap.read();acquired=time.perf_counter();read_ms=(acquired-t)*1000
            if not ok:raise RuntimeError('No camera frame.')
            if active and acquired-trials[-1]['start_time']>=8:
                trials[-1].update(end_time=acquired,end_frame=len(rows)-1,completed=True);active=False
            t=time.perf_counter();raw=cv2.flip(raw,1);mirror_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();detection=hands.process(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));mp_ms=(time.perf_counter()-t)*1000
            hand=detection.multi_hand_landmarks[0] if detection.multi_hand_landmarks else None
            result=model.update(hand);recognizer_end=time.perf_counter()
            t=time.perf_counter();view=raw.copy()
            if active:
                current=schedule[idx];prompt=f"Trial {idx+1}/24: {current['condition']} DIGIT {current['target']} | {8-(acquired-trials[-1]['start_time']):.1f}s"
            else:
                nxt=schedule[idx+1] if idx+1<len(schedule) else None
                prompt=f"NEXT digit {nxt['target']} ({nxt['condition']})" if nxt else 'ALL TRIALS RECORDED - Q to save'
            text(view,[prompt,'SPACE: begin, THEN start gesture. Do not pose target in advance.',f"Output: {result['stable'] or '?'} | {result['status']}",'Q / ESC / close window: save and exit'])
            cv2.imshow(window,view);submitted=time.perf_counter();display_ms=(submitted-t)*1000
            t=time.perf_counter();k=keycode(cv2.waitKey(1));events_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            if writer is None:
                h,w=raw.shape[:2];writer=cv2.VideoWriter(str(tmp/'video.avi'),cv2.VideoWriter_fourcc(*'MJPG'),25,(w,h))
                if not writer.isOpened():raise RuntimeError('Video encoder unavailable.')
            writer.write(raw);io_ms=(time.perf_counter()-t)*1000
            stage={k:result[k] for k in ['feature_ms','routing_ms','sequence_features_ms','inference_ms','smoothing_ms','recognizer_other_ms']}
            rows.append(dict(frame=len(rows),trial=idx+1,phase='active' if active else 'between',timestamp=acquired,display_submit_time=submitted,target=schedule[idx]['target'] if idx>=0 else '',condition=schedule[idx]['condition'] if idx>=0 else '',hand=int(hand is not None),sequence_size=result['sequence_size'],status=result['status'],inference_branch=result['inference_branch'],candidate=result['candidate'] or '',stable=result['stable'] or '',confidence=result['confidence'],read_ms=read_ms,mirror_ms=mirror_ms,mediapipe_ms=mp_ms,**stage,display_ms=display_ms,events_ms=events_ms,video_io_ms=io_ms,pipeline_to_submit_ms=(submitted-loop_start)*1000,acquired_to_recognizer_ms=(recognizer_end-acquired)*1000,iteration_ms=(time.perf_counter()-loop_start)*1000))
            closed=cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1
            if k in [27,ord('q')] or closed:break
            if k==32 and not active and idx+1<len(schedule):
                idx+=1
                if schedule[idx]['reset']:model.reset()
                trials.append(dict(trial=idx+1,**schedule[idx],start_frame=len(rows),start_time=time.perf_counter(),completed=False))
                annotations.append(dict(trial=idx+1,onset_frame='',execution='unreviewed'))
                active=True
        clean=True
    finally:
        if active and trials:trials[-1].update(end_time=time.perf_counter(),end_frame=len(rows)-1)
        cap.release();hands.close()
        if writer:writer.release()
        cv2.destroyAllWindows()
        if rows:write(session/'frames.csv',rows,list(rows[0]))
        save(session/'trials.json',trials);write(session/'onsets.csv',annotations,['trial','onset_frame','execution'])
        save(session/'recording_status.json',dict(clean_exit=clean,trials_started=len(trials),trials_completed=sum(t['completed'] for t in trials),frames=len(rows)))
        if (tmp/'video.avi').exists():shutil.copyfile(tmp/'video.avi',session/'recording.avi');(tmp/'video.avi').unlink();tmp.rmdir()
    print('RECORDING_SAVED',session,flush=True)

def review(session):
    rows=read(session/'frames.csv');trials=json.loads((session/'trials.json').read_text(encoding='utf-8'));annotations=read(session/'onsets.csv')
    trials=[t for t in trials if t.get('end_frame',-1)>=t['start_frame']]
    if not trials:raise RuntimeError('No recorded trial to review.')
    bytrial={int(a['trial']):a for a in annotations}
    tmp=tempdir();shutil.copyfile(session/'recording.avi',tmp/'video.avi');cap=cv2.VideoCapture(str(tmp/'video.avi'))
    n=0;frameidx=trials[0]['start_frame'];paused=True;window='Latency - mark gesture onset'
    try:
        while True:
            trial=trials[n];label=bytrial[trial['trial']]
            cap.set(cv2.CAP_PROP_POS_FRAMES,frameidx);ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode frame {frameidx}')
            # Do not show the classifier outputs during onset annotation.
            text(frame,[f"Trial {trial['trial']} / target {trial['target']} / {trial['condition']} / frame {frameidx}",f"Marked onset: {label['onset_frame']} | {label['execution']}",'O mark actual movement onset (valid execution); I wrong/invalid execution','SPACE play/pause; A/F previous/next frame; N/P next/previous trial','R clear mark; Q save. Keep a mark independent of system predictions.'])
            cv2.imshow(window,frame)
            k=keycode(cv2.waitKey(40 if not paused else 100))
            if k in [27,ord('q')] or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if k==ord('o'):label.update(onset_frame=frameidx,execution='valid');paused=True
            if k==ord('i'):label.update(onset_frame='',execution='invalid_execution');paused=True
            if k==ord('r'):label.update(onset_frame='',execution='unreviewed')
            if k==32:paused=not paused
            if k==ord('a'):frameidx=max(trial['start_frame'],frameidx-1);paused=True
            if k==ord('f'):frameidx=min(trial['end_frame'],frameidx+1);paused=True
            if k in [ord('n'),ord('p')]:
                n=min(len(trials)-1,n+1) if k==ord('n') else max(0,n-1)
                frameidx=trials[n]['start_frame'];paused=True;continue
            if not paused:
                if frameidx<trial['end_frame']:frameidx+=1
                else:paused=True
    finally:
        cap.release();cv2.destroyAllWindows();write(session/'onsets.csv',annotations,['trial','onset_frame','execution'])
        (tmp/'video.avi').unlink(missing_ok=True);tmp.rmdir()
    print('ONSETS_SAVED',session,flush=True)

def describe(values):
    a=np.array(values,dtype=float)
    if not len(a):return dict(n=0,mean=None,sample_sd=None,median=None,p95=None)
    return dict(n=len(a),mean=float(a.mean()),sample_sd=float(a.std(ddof=1)) if len(a)>1 else None,median=float(np.median(a)),p95=float(np.percentile(a,95)))

def trial_result(trial,annotation,rows):
    rr=[r for r in rows if int(r['trial'])==trial['trial'] and r['phase']=='active']
    outcome=dict(trial=trial['trial'],target=trial['target'],condition=trial['condition'],completed=trial['completed'],status='unreviewed',onset_frame='',output_frame='',latency_ms='',observed_after_onset_ms='',window_fill_ms='',candidate_to_stable_ms='')
    if rr and trial['reset']:
        valid=[r for r in rr if int(r['sequence_size'])>0]
        full=[r for r in valid if int(r['sequence_size'])==30]
        if full:
            first_full=full[0]
            starts=[r for r in valid if int(r['sequence_size'])==1 and int(r['frame'])<=int(first_full['frame'])]
            if starts:outcome['window_fill_ms']=1000*(float(first_full['timestamp'])-float(starts[-1]['timestamp']))
    if not trial['completed']:outcome['status']='incomplete';return outcome
    if annotation['execution']!='valid' or annotation['onset_frame']=='':outcome['status']=annotation['execution'];return outcome
    onset=int(annotation['onset_frame']);matched=[r for r in rr if int(r['frame'])==onset]
    if not matched:raise ValueError('Onset is outside the trial.')
    onsetrow=matched[0];outcome['onset_frame']=onset
    after=[r for r in rr if int(r['frame'])>=onset]
    outcome['observed_after_onset_ms']=(float(after[-1]['timestamp'])-float(onsetrow['timestamp']))*1000
    # Include inter-trial frames: an onset at the first active frame must not
    # turn an already displayed target into a spurious near-zero response.
    before=[r for r in rows if int(r['frame'])<onset]
    if before and str(before[-1]['stable'])==str(trial['target']):outcome['status']='already_correct_before_onset';return outcome
    correct=[r for r in after if str(r['stable'])==str(trial['target'])]
    if not correct:outcome['status']='no_correct_output_within_trial';return outcome
    output=correct[0];outcome.update(status='success',output_frame=output['frame'],latency_ms=1000*(float(output['display_submit_time'])-float(onsetrow['timestamp'])))
    # Last uninterrupted run of correct candidates ending at the stable output.
    run=[]
    for r in after:
        if str(r['candidate'])==str(trial['target']) and float(r['confidence'])>(.8 if trial['target']<10 else .9):run.append(r)
        else:run=[]
        if int(r['frame'])==int(output['frame']):break
    if run:outcome['candidate_to_stable_ms']=1000*(float(output['display_submit_time'])-float(run[0]['display_submit_time']))
    return outcome

def analyze(session):
    rows=read(session/'frames.csv');annotations={int(a['trial']):a for a in read(session/'onsets.csv')}
    trials=json.loads((session/'trials.json').read_text(encoding='utf-8'))
    outcomes=[trial_result(t,annotations[t['trial']],rows) for t in trials]
    fields=['trial','target','condition','completed','status','onset_frame','output_frame','latency_ms','observed_after_onset_ms','window_fill_ms','candidate_to_stable_ms']
    write(session/'trial_latency.csv',outcomes,fields)
    stages=['read_ms','mirror_ms','mediapipe_ms','feature_ms','routing_ms','sequence_features_ms','inference_ms','smoothing_ms','recognizer_other_ms','display_ms','events_ms','video_io_ms','pipeline_to_submit_ms','iteration_ms']
    active=[r for r in rows if r['phase']=='active']; stage_rows=[]
    for branch in ['all','static','dynamic']:
        subset=active if branch=='all' else [r for r in active if r['inference_branch']==branch]
        for s in stages:
            # Empty-route/warmup frames do not become zero-latency inference calls.
            selected=subset
            if s in ['inference_ms','smoothing_ms']:selected=[r for r in subset if r['inference_branch']]
            elif s in ['feature_ms','routing_ms']:selected=[r for r in subset if int(r['hand'])==1]
            elif s=='sequence_features_ms':selected=[r for r in subset if r['inference_branch']=='dynamic']
            stage_rows.append(dict(branch=branch,stage=s,**describe([float(r[s]) for r in selected])))
    write(session/'stage_summary.csv',stage_rows,['branch','stage','n','mean','sample_sd','median','p95'])
    groups={}
    for condition in ['reset_static','reset_dynamic','continuous_transition']:
        subset=[r for r in outcomes if r['condition']==condition]
        groups[condition]=dict(started=len(subset),statuses={status:sum(r['status']==status for r in subset) for status in sorted({r['status'] for r in subset})},latency_ms_success_only=describe([r['latency_ms'] for r in subset if r['status']=='success']))
    intervals=np.diff([float(r['timestamp']) for r in rows])
    fps=float(1/np.mean(intervals)) if len(intervals) and np.mean(intervals)>0 else None
    save(session/'latency_report.json',dict(analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),conditions=groups,measured_acquisition_loop_fps=fps,scope='camera-acquired visible onset to correct stable software display submission; not photon-to-photon physical latency',failure_policy='no output within trial is reported separately, not silently excluded; means describe successes only',window_policy='sliding30, no post-prediction clear; reset conditions clear state before trial, continuous conditions keep preceding state',timing_policy='sum of frame processing stages is not gesture latency; sequence accumulation overlaps frame processing; recording/UI instrumentation included in observed cadence',source='live recorded timestamps, manually labelled video onsets'))
    content=['# 实时响应测量结果','','端点：录像中人工标记的动作开始帧被读入程序，到首次稳定正确结果提交给显示接口。未测量摄像头曝光/传输及显示器物理刷新延迟。','','| 条件 | 已开始次数 | 成功次数 | 成功响应均值(ms) | SD(ms) | P95(ms) |','|---|---:|---:|---:|---:|---:|']
    for condition,g in groups.items():
        s=g['latency_ms_success_only'];fmt=lambda v:'—' if v is None else f'{v:.2f}'
        content.append(f"| {condition} | {g['started']} | {s['n']} | {fmt(s['mean'])} | {fmt(s['sample_sd'])} | {fmt(s['p95'])} |")
    content+=['','完整失败、未标注、执行无效及不完整次数见latency_report.json；不能只报告成功次数。逐阶段均值/标准差/P95见stage_summary.csv，逐动作窗口填充和输出时间见trial_latency.csv。','','窗口填充时间按本次重置后第一条有效手部帧至第30条有效手部帧计算，不等于动作起点后的30条纯动作帧。candidate_to_stable_ms为末次连续正确高置信候选到稳定输出的观测间隔，并非额外一次平滑函数的计算耗时。','','测试基于记录在protocol.json中的修复版与权重，不等同于复现旧稿22FPS/45ms；只有人工标注和有效性核查完成后才可用于回复审稿。']
    (session/'results.md').write_text('\n'.join(content)+'\n',encoding='utf-8')
    print('ANALYSIS_SAVED',session,flush=True)

def check():
    # Compare state/outputs to the original class using deterministic mock classifiers.
    class Dummy:
        def __init__(self,index,n):self.index=index;self.n=n
        def __call__(self,x,training=False):
            p=np.zeros((1,self.n),np.float32);p[0,self.index]=.99;p[0,(self.index+1)%self.n]=.01
            return tf.constant(p)
    from types import SimpleNamespace
    base=Recognizer.__new__(Recognizer);timed=TimedRecognizer.__new__(TimedRecognizer)
    for m in [base,timed]:m.static=Dummy(1,10);m.dynamic=Dummy(2,11);m.reset()
    rng=np.random.RandomState(42);points=rng.uniform(0,1,(21,3))
    for i in range(70):
        p=points if i<35 else rng.uniform(0,1,(21,3))
        hand=None if i==69 else SimpleNamespace(landmark=[SimpleNamespace(x=float(x),y=float(y),z=float(z),visibility=0.) for x,y,z in p])
        a=base.update(hand);b=timed.update(hand)
        for k in ['status','variance','candidate','stable','confidence']:assert a[k]==b[k],(i,k,a[k],b[k])
        assert all(b[k]>=0 for k in ['feature_ms','routing_ms','sequence_features_ms','inference_ms','smoothing_ms','recognizer_other_ms'])
    # Exercise actual supplied weights, including a full dynamic window, offline.
    base=Recognizer();timed=TimedRecognizer()
    for i in range(42):
        p=points if i<15 else rng.uniform(0,1,(21,3))
        hand=SimpleNamespace(landmark=[SimpleNamespace(x=float(x),y=float(y),z=float(z),visibility=0.) for x,y,z in p])
        a=base.update(hand);b=timed.update(hand)
        for k in ['status','variance','candidate','stable','confidence']:assert a[k]==b[k],('real_weights',i,k)
    tmp=tempdir()
    try:
        writer=cv2.VideoWriter(str(tmp/'v.avi'),cv2.VideoWriter_fourcc(*'MJPG'),25,(320,240));assert writer.isOpened()
        writer.write(np.zeros((240,320,3),np.uint8));writer.release();cap=cv2.VideoCapture(str(tmp/'v.avi'));ok,_=cap.read();cap.release();assert ok
        rows=[dict(frame=i,trial=1,phase='active',hand=1,timestamp=100+i*.05,display_submit_time=100+i*.05+.02,sequence_size=i+1,candidate='1',confidence=.99,stable='1' if i>=3 else '',inference_branch='static',**{s:1. for s in ['read_ms','mirror_ms','mediapipe_ms','feature_ms','routing_ms','sequence_features_ms','inference_ms','smoothing_ms','recognizer_other_ms','display_ms','events_ms','video_io_ms','pipeline_to_submit_ms','iteration_ms']}) for i in range(6)]
        trial=dict(trial=1,target=1,condition='reset_static',reset=True,completed=True)
        ann=dict(trial=1,onset_frame=1,execution='valid')
        result=trial_result(trial,ann,rows);assert result['status']=='success' and abs(result['latency_ms']-120)<1e-6
        failed=[dict(r,stable='') for r in rows];assert trial_result(trial,ann,failed)['status']=='no_correct_output_within_trial'
        assert trial_result(dict(trial,completed=False),ann,rows)['status']=='incomplete'
        assert trial_result(trial,dict(ann,onset_frame=4),rows)['status']=='already_correct_before_onset'
        preexisting=[dict(r,phase='between' if i==0 else 'active',stable='1') for i,r in enumerate(rows)]
        assert trial_result(trial,ann,preexisting)['status']=='already_correct_before_onset'
        reset_rows=[dict(r,frame=i,sequence_size=s,timestamp=100+i*.05) for i,s in enumerate([1,2,0,1,2,30]) for r in [rows[min(i,len(rows)-1)]]]
        assert abs(trial_result(trial,ann,reset_rows)['window_fill_ms']-100)<1e-6
        write(tmp/'frames.csv',rows,list(rows[0]));write(tmp/'onsets.csv',[ann],['trial','onset_frame','execution']);save(tmp/'trials.json',[trial]);analyze(tmp)
        summary=json.loads((tmp/'latency_report.json').read_text(encoding='utf-8'));assert summary['conditions']['reset_static']['latency_ms_success_only']['n']==1
        save(ROOT/'offline_check.json',dict(passed=True,camera_opened=False,mock_and_real_weights_parity=True,video_codec=True,latency_arithmetic=True,window_reset_arithmetic=True,failed_incomplete_and_preexisting_output=True,report_generation=True,scope='offline implementation checks only, not measured camera experiment results'))
        print('OFFLINE_CHECK_PASSED: real/mock parity, video codec, timestamps, failed/incomplete trials, reports. NO CAMERA OPENED.',flush=True)
    finally:
        for p in tmp.iterdir():p.unlink()
        tmp.rmdir()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['record','review','analyze','check']);p.add_argument('--session',type=Path);args=p.parse_args()
    if args.mode=='record':record()
    elif args.mode=='check':check()
    elif args.mode=='review':review(args.session or latest())
    else:analyze(args.session or latest())
