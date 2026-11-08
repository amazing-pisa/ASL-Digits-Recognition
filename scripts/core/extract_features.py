"""Extract 30x212 features from an optional user-provided video. No frames are saved."""
import runtime
import argparse
from pathlib import Path
import cv2, numpy as np, mediapipe as mp
def extract_keypoints_from_video(video_path, seq_len=30):
    cap = cv2.VideoCapture(video_path)
    keypoints_seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(img_rgb)
        if not results.multi_hand_landmarks:
            continue
        res = results.multi_hand_landmarks[0]
        joint = np.zeros((21, 4))
        base = [res.landmark[0].x, res.landmark[0].y, res.landmark[0].z]
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x - base[0], lm.y - base[1], lm.z - base[2], lm.visibility]
        max_dist = np.max(np.linalg.norm(joint[:, :3], axis=1))
        if max_dist == 0:
            max_dist = 1e-05
        joint[:, :3] /= max_dist
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
        v = (v2 - v1) / (np.linalg.norm(v2 - v1, axis=1)[:, np.newaxis] + 1e-05)
        angle = np.degrees(np.arccos(np.clip(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :], v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]), -1.0, 1.0)))
        pinch_20 = np.linalg.norm(joint[8, :3] - joint[4, :3])
        index_to_wrist = np.linalg.norm(joint[8, :3] - joint[0, :3])
        middle_to_wrist = np.linalg.norm(joint[12, :3] - joint[0, :3])
        ring_to_wrist = np.linalg.norm(joint[16, :3] - joint[0, :3])
        pinky_to_wrist = np.linalg.norm(joint[20, :3] - joint[0, :3])
        wrist_twist_z = joint[5, 2] - joint[17, 2]
        wrist_twist_x = np.linalg.norm(joint[5, :2] - joint[17, :2])
        asl_features = [pinch_20, index_to_wrist, middle_to_wrist, ring_to_wrist, pinky_to_wrist, wrist_twist_z, wrist_twist_x]
        frame_keypoints = np.concatenate([joint.flatten(), angle, asl_features])
        keypoints_seq.append(frame_keypoints)
    cap.release()
    video_length = len(keypoints_seq)
    if video_length == 0:
        return None
    keypoints_seq = np.array(keypoints_seq, dtype=np.float32)
    delta_seq = np.zeros_like(keypoints_seq)
    if video_length > 1:
        delta_seq[1:] = keypoints_seq[1:] - keypoints_seq[:-1]
    enhanced_seq = np.concatenate([keypoints_seq, delta_seq], axis=-1)
    if video_length < seq_len:
        pad_len = seq_len - video_length
        last_frame = enhanced_seq[-1].copy()
        last_frame[106:] = 0.0
        pad_matrix = np.tile(last_frame, (pad_len, 1))
        final_seq = np.vstack([enhanced_seq, pad_matrix])
    elif video_length > seq_len:
        indices = np.linspace(0, video_length - 1, seq_len, dtype=int)
        final_seq = enhanced_seq[indices]
    else:
        final_seq = enhanced_seq
    return final_seq
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('video');parser.add_argument('output');parser.add_argument('--digit',type=int,required=True,choices=range(10,21));args=parser.parse_args()
    hands_model=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.5,min_tracking_confidence=.5)
    try: x=extract_keypoints_from_video(args.video)
    finally:hands_model.close()
    if x is None:raise SystemExit('No valid hand landmarks found')
    y=np.full((30,1),args.digit-10,dtype=np.float32)
    np.save(args.output,np.concatenate([x,y],axis=1)[None])
