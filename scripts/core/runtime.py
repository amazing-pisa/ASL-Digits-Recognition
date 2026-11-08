"""Repository-relative paths and CPU runtime configuration."""
from pathlib import Path
import json, os
import sys, importlib.util, hashlib, shutil, tempfile
ROOT=Path(__file__).resolve().parent
REPOSITORY=ROOT.parents[1]
sys.path.insert(0,str(REPOSITORY/'models'))
ARTIFACT_ROOT=Path(os.environ.get('ASL_ARTIFACT_ROOT',str(REPOSITORY))).resolve()
CONFIG=json.loads((REPOSITORY/'configs/system.json').read_text(encoding='utf-8'))
SOURCE=Path(os.environ.get('ASL_DATA_ROOT',str(ARTIFACT_ROOT/'data'))).resolve()
STATIC_DATASET=SOURCE/'Sign-Language-Digits-Dataset'
DYNAMIC_DATASET=SOURCE/'dataset'
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','6')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('MPLBACKEND','Agg')
# MediaPipe's Windows native loader requires an ASCII package-resource path.
if sys.platform=='win32':
    spec=importlib.util.find_spec('mediapipe')
    if spec and spec.origin:
        package=Path(spec.origin).parent
        if not str(package).isascii():
            key=hashlib.sha256((str(package)+str((package/'__init__.py').stat().st_mtime_ns)).encode()).hexdigest()[:10]
            cached=Path(tempfile.gettempdir())/('asl_mediapipe_'+key)
            if not (cached/'ready.txt').exists():
                cached.mkdir(exist_ok=True)
                shutil.copytree(package,cached/'mediapipe',dirs_exist_ok=True)
                (cached/'ready.txt').write_text('cached package resources',encoding='utf-8')
            sys.path.insert(0,str(cached))
(ROOT/'models').mkdir(exist_ok=True)
(ROOT/'cache').mkdir(exist_ok=True)
MODEL_FILES={'cnn':'hand_sign_cnn_opt_model.h5','fnn':'paper_fnn_model.h5','vgg':'full_vgg16_fusion_model.h5','stcnn':'best_st_cnn_opt.h5','glt':'GLT.h5','lite':'best_transformer_lite.h5'}
def model_path(kind):
    name=MODEL_FILES[kind]
    for p in [ROOT/'models'/name.replace('.h5','.keras'),ROOT/'models'/name,ARTIFACT_ROOT/'weights/baseline'/name]:
        if p.is_file():return str(p)
    raise FileNotFoundError('Install the weights artifact in weights/baseline: '+name)
def dynamic_test_dir():
    path=os.environ.get('ASL_RAW_TEST_VIDEOS','')
    if not path or not Path(path).is_dir():raise FileNotFoundError('Raw videos are not distributed. Use reproduce.py evaluate for released feature sequences.')
    return path
