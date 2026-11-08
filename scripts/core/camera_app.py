import runtime
import argparse
from collections import deque
import numpy as np
import cv2
import mediapipe as mp
from compat import normalize_points,load_model_compat


def extract_features(res):
    coords=np.array([[p.x,p.y,p.z] for p in res.landmark],dtype=np.float32)
    normalized=normalize_points(coords)
    joint=np.column_stack([normalized,[getattr(p,'visibility',0.) for p in res.landmark]])
    a=[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
    b=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    v=normalized[b]-normalized[a]
    v=v/(np.linalg.norm(v,axis=1)[:,None]+1e-5)
    i=[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]
    j=[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
    angles=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',v[i],v[j]),-1,1)))
    special=[np.linalg.norm(normalized[8]-normalized[4]),*[np.linalg.norm(normalized[k]-normalized[0]) for k in [8,12,16,20]],normalized[5,2]-normalized[17,2],np.linalg.norm(normalized[5,:2]-normalized[17,:2])]
    base=np.concatenate([joint.flatten(),angles,special]).astype(np.float32)
    if base.shape!=(106,) or not np.isfinite(base).all():raise ValueError('Invalid landmarks/features')
    return normalized,base


class Recognizer:
    def __init__(self,static_only=False):
        self.static=load_model_compat(runtime.model_path('cnn'))
        self.dynamic=None if static_only else load_model_compat(runtime.model_path('stcnn'))
        self.reset()
    def reset(self):
        self.joints=deque(maxlen=10)
        self.sequence=deque(maxlen=30)
        self.history=deque(maxlen=3)
        self.previous_branch=None
    def update(self,res):
        if res is None:
            self.reset()
            return {'status':'No hand','variance':None,'candidate':None,'stable':None,'confidence':0.}
        points,base=extract_features(res)
        self.joints.append(points)
        self.sequence.append(base)
        result={'status':'Warming up','variance':None,'candidate':None,'stable':None,'confidence':0.}
        if len(self.joints)<10:return result
        variance=float(np.var(np.asarray(self.joints),axis=0).mean())
        branch='Static' if self.dynamic is None or variance<float(runtime.CONFIG['motion_threshold']) else 'Dynamic'
        result.update(status=branch,variance=variance)
        if branch!=self.previous_branch:self.history.clear()
        self.previous_branch=branch
        if branch=='Static':
            probability=self.static(points[None,:,:],training=False).numpy()[0]
            threshold=.8;offset=0
        else:
            if len(self.sequence)<30:
                result['status']='Accumulating dynamic frames';self.history.clear();return result
            bases=np.asarray(self.sequence,dtype=np.float32)
            delta=np.zeros_like(bases);delta[1:]=bases[1:]-bases[:-1]
            probability=self.dynamic(np.concatenate([bases,delta],axis=-1)[None,:,:],training=False).numpy()[0]
            threshold=.9;offset=10
        index=int(probability.argmax());confidence=float(probability[index]);candidate=str(index+offset)
        result.update(candidate=candidate,confidence=confidence)
        if confidence>threshold:self.history.append(candidate)
        else:self.history.clear()
        if len(self.history)==3 and len(set(self.history))==1:result['stable']=candidate
        return result


def main(static_only=False):
    p=argparse.ArgumentParser()
    p.add_argument('--camera',type=int,default=runtime.CONFIG.get('camera_index',0))
    p.add_argument('--static-only',action='store_true',default=static_only)
    args=p.parse_args()
    recognizer=Recognizer(args.static_only)
    hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.5,min_tracking_confidence=.5)
    cap=cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():raise RuntimeError(f'Cannot open camera {args.camera}. Close other camera apps or change --camera / configs/system.json.')
        print('Recognition started. Press q or Esc to exit.',flush=True)
        while True:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera opened but no frame received.')
            frame=cv2.flip(frame,1)
            detected=hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            hand=detected.multi_hand_landmarks[0] if detected.multi_hand_landmarks else None
            result=recognizer.update(hand)
            if hand:mp.solutions.drawing_utils.draw_landmarks(frame,hand,mp.solutions.hands.HAND_CONNECTIONS)
            lines=[f"Status: {result['status']}",f"Prediction: {result['stable'] or '?'}  Confidence: {result['confidence']:.2f}"]
            if result['variance'] is not None:lines.append(f"Motion variance: {result['variance']:.6f}")
            for i,line in enumerate(lines):cv2.putText(frame,line,(10,32+34*i),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
            cv2.imshow('ASL 0-20 repaired',frame)
            if cv2.waitKey(1)&0xff in [27,ord('q')]:break
    finally:
        cap.release();hands.close();cv2.destroyAllWindows()


if __name__=='__main__':main()
