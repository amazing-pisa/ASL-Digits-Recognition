"""Record routing trials, annotate independently, then evaluate a fixed threshold."""
from pathlib import Path
import sys, csv, json, time, random, tempfile, shutil, argparse, uuid, hashlib
from datetime import datetime
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / 'scripts/core'))
import runtime
from camera_app import Recognizer, cv2, mp
import numpy as np

def temporary_directory(prefix):
    # Default mkdir permissions avoid the restricted Windows mode-700 temp bug.
    p=Path(tempfile.gettempdir())/(prefix+uuid.uuid4().hex)
    p.mkdir()
    return p

def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def writecsv(path, rows, fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def lines(frame, text):
    for i,t in enumerate(text):
        cv2.putText(frame,t,(12,28+29*i),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,0),2)

def latest():
    sessions=sorted((ROOT/'sessions').glob('*/frames.csv'))
    if not sessions: raise RuntimeError('No recording found. Run record first.')
    return sessions[-1].parent

def record(args):
    session=ROOT/'sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True)
    schedule=[dict(digit=d,style=('hold' if d<10 else ('normal' if r==0 else 'slow_pause'))) for r in range(2) for d in range(21)]
    random.Random(42).shuffle(schedule)
    save(session/'schedule.json',schedule)
    hashes={name:hashlib.sha256(((runtime.REPOSITORY/'models'/name) if name=='model_builders.py' else Path(runtime.__file__).parent/name).read_bytes()).hexdigest() for name in ['camera_app.py','runtime.py','compat.py','model_builders.py']}
    hashes['configs/system.json']=hashlib.sha256((runtime.REPOSITORY/'configs/system.json').read_bytes()).hexdigest()
    hashes['motion_test.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for kind in ['cnn','stcnn']:
        p=Path(runtime.model_path(kind));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(session/'protocol.json',dict(threshold=float(runtime.CONFIG['motion_threshold']),motion_window=10,sequence_window=30,planned_trials=len(schedule),trial_seconds=6,schedule_seed=42,source='scripts/core/camera_app.py',sha256=hashes,python=sys.version,opencv=cv2.__version__,mediapipe=mp.__version__,tensorflow=__import__('tensorflow').__version__,no_hand_policy='reset',video_fps_for_container_only=25,notes='Truth must be independently annotated. Digit class is not motion-state truth. Original timestamps are in frames.csv. Fixed threshold; not a threshold-selection experiment.'))
    recognizer=Recognizer()
    hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.5,min_tracking_confidence=.5)
    cap=cv2.VideoCapture(runtime.CONFIG.get('camera_index',0))
    temp=temporary_directory('cmc_motion_')
    writer=None; rows=[]; trial=-1; active=False; start=0.; done=False
    print('SPACE: start next 6-second trial. Q: finish. S/D ground truth is assigned later, not during recording.',flush=True)
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot open camera. Close other camera apps.')
        while True:
            ok,raw=cap.read()
            if not ok: raise RuntimeError('No camera frame received.')
            raw=cv2.flip(raw,1)
            timestamp=time.perf_counter()
            if writer is None:
                h,w=raw.shape[:2];writer=cv2.VideoWriter(str(temp/'recording.avi'),cv2.VideoWriter_fourcc(*'MJPG'),25,(w,h))
                if not writer.isOpened(): raise RuntimeError('Video encoder could not be opened.')
            if active and timestamp-start>=6: active=False
            tic=time.perf_counter()
            found=hands.process(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
            hand=found.multi_hand_landmarks[0] if found.multi_hand_landmarks else None
            result=recognizer.update(hand)
            processing_ms=(time.perf_counter()-tic)*1000
            score=result['variance']
            route='' if score is None else ('static' if score<float(runtime.CONFIG['motion_threshold']) else 'dynamic')
            task=schedule[trial] if trial>=0 else dict(digit='',style='ready')
            writer.write(raw)
            rows.append(dict(frame=len(rows),timestamp=timestamp,trial=trial+1,phase='active' if active else 'between',target_digit=task['digit'],style=task['style'],hand_detected=int(hand is not None),motion_score='' if score is None else score,route=route,candidate=result['candidate'] or '',displayed=result['stable'] or '',confidence=result['confidence'],processing_ms=processing_ms))
            view=raw.copy()
            next_task=schedule[trial+1] if trial+1<len(schedule) else None
            if active:
                prompt=f"Trial {trial+1}/42: digit {task['digit']}  {task['style']}  {max(0,6-(timestamp-start)):.1f}s"
            else:
                prompt=f"NEXT: {next_task['digit']} {next_task['style']} - SPACE when ready" if next_task else 'All trials recorded. Q to finish.'
            lines(view,[prompt,'hold: keep pose still; normal: repeat gesture','slow_pause: move slowly, pause briefly, then continue','SPACE next trial | Q finish. Recording contains no route overlay.'])
            cv2.imshow('Motion study - record',view)
            key=cv2.waitKey(1)&255
            if key in [27,ord('q')]: break
            if key==32 and not active and next_task:
                trial+=1;active=True;start=time.perf_counter()
                # Keep continuous state across trials to retain actual transition behavior.
        done=True
    finally:
        cap.release();hands.close()
        if writer: writer.release()
        cv2.destroyAllWindows()
        if rows:
            writecsv(session/'frames.csv',rows,list(rows[0]))
            if (temp/'recording.avi').exists(): shutil.copyfile(temp/'recording.avi',session/'recording.avi')
            writecsv(session/'annotations.csv',[dict(frame=r['frame'],motion_truth='unlabelled') for r in rows],['frame','motion_truth'])
        save(session/'recording_status.json',dict(clean_exit=done,frames=len(rows),trials_started=trial+1,last_trial_completed=not active))
        # No recursive deletion; preserve temporary capture if final copying fails.
        if (session/'recording.avi').exists():
            (temp/'recording.avi').unlink(missing_ok=True)
            temp.rmdir()
    print('RECORDING_SAVED',session,flush=True)

def review(session):
    rows=readcsv(session/'frames.csv')
    annotations=readcsv(session/'annotations.csv')
    assert len(rows)==len(annotations)
    # Video backends can fail on Chinese paths; read an ASCII temporary copy.
    temp=temporary_directory('cmc_review_')
    shutil.copyfile(session/'recording.avi',temp/'video.avi')
    cap=cv2.VideoCapture(str(temp/'video.avi'))
    index=0; paused=True; label=None; frames_seen=set()
    print('S = static/no intended motion, D = intended motion, U = ambiguous/unlabelled. SPACE play/pause; A back 1; F forward 1; Q save.',flush=True)
    try:
        while index<len(rows):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Cannot read video frame {index}')
            r=rows[index]
            lines(frame,[f"Frame {index}/{len(rows)-1} | trial {r['trial']} | digit {r['target_digit']} | {r['phase']}",f"Label: {annotations[index]['motion_truth']}  Paint: {label}",'S static | D dynamic | U unsure | SPACE play/pause','A back one | F forward one | Q save and exit'])
            cv2.imshow('Motion study - independent annotation',frame)
            key=cv2.waitKey(40 if not paused else 100)&255
            if key in [27,ord('q')]: break
            if key in [ord('s'),ord('d'),ord('u')]:
                label={ord('s'):'static',ord('d'):'dynamic',ord('u'):'unlabelled'}[key]
                annotations[index]['motion_truth']=label;frames_seen.add(index)
            if key==32: paused=not paused
            if key==ord('a'):
                index=max(0,index-1);paused=True;label=None;continue
            if key==ord('f'):
                index=min(len(rows)-1,index+1);paused=True;continue
            if not paused:
                if label is not None: annotations[index]['motion_truth']=label
                frames_seen.add(index);index+=1
    finally:
        cap.release();cv2.destroyAllWindows()
        writecsv(session/'annotations.csv',annotations,['frame','motion_truth'])
        save(session/'annotation_status.json',dict(frames_visited_this_session=len(frames_seen),labelled_frames=sum(r['motion_truth']!='unlabelled' for r in annotations),total_frames=len(rows)))
        (temp/'video.avi').unlink(missing_ok=True);temp.rmdir()
    print('ANNOTATIONS_SAVED',session,flush=True)

def analyze(session):
    rows=readcsv(session/'frames.csv'); labels=readcsv(session/'annotations.csv')
    assert len(rows)==len(labels) and all(r['frame']==a['frame'] for r,a in zip(rows,labels))
    cm={t:{p:0 for p in ['static','dynamic']} for t in ['static','dynamic']}
    trial={};annotated=0;active=0;missing=0;warmup=0;uncertain=0
    for r,a in zip(rows,labels):
        if r['phase']!='active':continue
        active+=1;t=a['motion_truth']
        if t not in cm:uncertain+=1;continue
        annotated+=1
        if r['hand_detected']!='1':missing+=1;continue
        if r['route'] not in cm[t]:warmup+=1;continue
        pred=r['route'];cm[t][pred]+=1
        item=trial.setdefault(r['trial'],dict(trial=r['trial'],target_digit=r['target_digit'],style=r['style'],scored=0,correct=0))
        item['scored']+=1;item['correct']+=int(t==pred)
    scored=sum(sum(v.values()) for v in cm.values()); correct=cm['static']['static']+cm['dynamic']['dynamic']
    if not scored: raise RuntimeError('No labelled active frames with valid routes. Complete annotation first.')
    report=dict(evaluation_unit='frame; correlated within trials, not independent action repetitions',threshold=json.loads((session/'protocol.json').read_text(encoding='utf-8'))['threshold'],active_frames=active,annotated_frames=annotated,unlabelled_active_frames=uncertain,missing_hand_among_annotated=missing,warmup_among_annotated=warmup,scored_frames=scored,confusion_matrix=cm,conditional_accuracy=correct/scored,scored_coverage_of_annotated=scored/annotated,static_false_dynamic_rate=cm['static']['dynamic']/sum(cm['static'].values()) if sum(cm['static'].values()) else None,dynamic_false_static_rate=cm['dynamic']['static']/sum(cm['dynamic'].values()) if sum(cm['dynamic'].values()) else None,notes='Accuracy is conditional on annotated frames with a valid route. Excluded frames and coverage must be reported. No claim of 200 independent trials, 97%, final digit accuracy, threshold optimum, or end-to-end delay. Inspect trial balance, annotation and videos before manuscript use.')
    save(session/'routing_report.json',report)
    writecsv(session/'confusion_matrix.csv',[dict(truth=t,**v) for t,v in cm.items()],['truth','static','dynamic'])
    for item in trial.values():item['conditional_accuracy']=item['correct']/item['scored']
    writecsv(session/'per_trial.csv',list(trial.values()),['trial','target_digit','style','scored','correct','conditional_accuracy'])
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

def check():
    # No camera access. Exercise matrix, exclusions and video codec roundtrip.
    temp=temporary_directory('cmc_motion_check_')
    try:
        rows=[];labels=[]
        for i,(truth,pred,hand) in enumerate([('static','static',1),('static','dynamic',1),('dynamic','static',1),('dynamic','dynamic',1),('dynamic','',0),('static','',1),('unlabelled','static',1)]):
            rows.append(dict(frame=i,phase='active',trial=1,target_digit=10,style='test',route=pred,hand_detected=hand))
            labels.append(dict(frame=i,motion_truth=truth))
        writecsv(temp/'frames.csv',rows,list(rows[0]));writecsv(temp/'annotations.csv',labels,list(labels[0]));save(temp/'protocol.json',dict(threshold=.0005))
        analyze(temp)
        r=json.loads((temp/'routing_report.json').read_text(encoding='utf-8'))
        assert r['conditional_accuracy']==.5 and r['scored_frames']==4 and r['missing_hand_among_annotated']==1 and r['warmup_among_annotated']==1 and r['unlabelled_active_frames']==1
        writer=cv2.VideoWriter(str(temp/'video.avi'),cv2.VideoWriter_fourcc(*'MJPG'),25,(320,240))
        assert writer.isOpened();writer.write(np.zeros((240,320,3),dtype=np.uint8));writer.release()
        cap=cv2.VideoCapture(str(temp/'video.avi'));ok,frame=cap.read();cap.release();assert ok and frame.shape==(240,320,3)
        print('OFFLINE_CHECK_PASSED_NO_CAMERA_OPENED',flush=True)
    finally:
        for p in temp.iterdir():p.unlink()
        temp.rmdir()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['record','review','analyze','check']);p.add_argument('--session',type=Path)
    args=p.parse_args()
    if args.mode=='record':record(args)
    elif args.mode=='check':check()
    elif args.mode=='review':review(args.session or latest())
    else:analyze(args.session or latest())
