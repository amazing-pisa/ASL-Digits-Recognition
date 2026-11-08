import runtime
import json
from pathlib import Path
import numpy as np
import cv2
from model_builders import build_model
from keras.src.saving import saving_lib


def _keras_asset_directory():
    """Avoid Python's private 0700 temp directories under a restricted Windows token."""
    import uuid
    p=runtime.ROOT/'cache'/'temporary_assets'/uuid.uuid4().hex
    p.mkdir(parents=True)
    return str(p)


# Keras 3.4.1 creates an assets directory even when a model has no custom assets.
saving_lib.get_temp_dir=_keras_asset_directory


def read_image(path, flags=cv2.IMREAD_COLOR):
    """OpenCV imread on Windows does not reliably accept Unicode paths."""
    try:
        raw=np.fromfile(str(path),dtype=np.uint8)
        return cv2.imdecode(raw,flags) if raw.size else None
    except (OSError,ValueError):
        return None


def normalize_points(points):
    p=np.asarray(points,dtype=np.float32)
    p=p-p[0]
    scale=float(np.max(np.linalg.norm(p,axis=1)))
    return p/scale if scale>0 else p


def normalize_fnn_batch(x):
    a=np.asarray(x,dtype=np.float32).reshape(-1,21,3)[:,:,:2].copy()
    a-=a[:,:1,:]
    scale=np.linalg.norm(a,axis=-1).max(axis=1)
    a/=np.where(scale>0,scale,1)[:,None,None]
    return a.reshape(-1,42)


def initialize_vgg_backbone(base):
    """Read the frozen backbone from the supplied model, without downloading ImageNet."""
    import h5py
    with h5py.File(Path(runtime.model_path('vgg')),'r') as f:
        for layer in base.layers:
            if not layer.weights:continue
            group=f['model_weights'][layer.name]
            names=[n.decode() if isinstance(n,bytes) else n for n in group.attrs['weight_names']]
            layer.set_weights([group[n][()] for n in names])


def load_model_compat(path,*args,**kwargs):
    name=Path(str(path)).name.lower()
    if 'fnn' in name:kind='fnn'
    elif 'vgg' in name:kind='vgg'
    elif 'transformer' in name:kind='lite'
    elif 'glt' in name:kind='glt'
    elif 'st_cnn' in name or 'stcnn' in name:kind='stcnn'
    elif 'cnn' in name:kind='cnn'
    else:raise ValueError(f'Unsupported model name: {path}')
    model=build_model(kind)
    actual=Path(path)
    if not actual.is_file():actual=Path(runtime.model_path(kind))
    model.load_weights(str(actual))
    return model


SCALER=runtime.ROOT/'cache/vgg_training_scaler.json'


def save_scaler(scaler):
    SCALER.write_text(json.dumps({'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'source':'fitted_during_training'},ensure_ascii=False,indent=2),encoding='utf-8')


def standardize_vgg(x):
    if not SCALER.exists():
        import mediapipe as mp
        from sklearn.preprocessing import StandardScaler
        features=[]
        paths=sorted(p for p in (runtime.STATIC_DATASET/'train').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg'])
        print('[Info] Rebuilding VGG scaler from TRAIN images only:',len(paths),flush=True)
        with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1) as hands:
            for i,p in enumerate(paths):
                img=read_image(p)
                if img is not None:
                    result=hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    if result.multi_hand_landmarks:
                        features.append(np.array([[v.x,v.y,v.z] for v in result.multi_hand_landmarks[0].landmark]).flatten())
                if (i+1)%500==0:print('[Info] Training scaler images processed:',i+1,flush=True)
        if not features:raise RuntimeError('No training hand landmarks found for VGG scaler')
        scaler=StandardScaler().fit(np.asarray(features,dtype=np.float32))
        save_scaler(scaler)
        d=json.loads(SCALER.read_text(encoding='utf-8'));d.update(source='reconstructed_from_current_training_images',images_with_landmarks=len(features))
        SCALER.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    d=json.loads(SCALER.read_text(encoding='utf-8'))
    return (np.asarray(x,dtype=np.float32)-np.asarray(d['mean'],dtype=np.float32))/np.asarray(d['scale'],dtype=np.float32)
