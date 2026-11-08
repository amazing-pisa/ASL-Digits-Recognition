"""Run an isolated, paired-seed, parameter-matched Delta ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE/'scripts/core'))
import runtime
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '6')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import numpy as np
import sklearn
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras import layers as L

OUT = HERE / 'outputs'
SOURCE = runtime.SOURCE
VARIANTS = {'full_delta': (212, 128), 'no_delta_original': (106, 128), 'no_delta_matched': (106, 377)}
EXPECTED = {'full_delta': 275403, 'no_delta_original': 207563, 'no_delta_matched': 275291}
SEEDS = [2021, 2022, 2023]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(input_dim, head_units):
    inputs = L.Input(shape=(30, input_dim))
    joints = L.Lambda(lambda x: x[:, :, :84])(inputs)
    joints = L.Reshape((30, 21, 4))(joints)
    x = L.Conv2D(64, (1, 3), activation='relu', padding='same')(joints)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(128, (3, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Lambda(lambda x: tf.reduce_mean(x, axis=2))(x)
    other = L.Lambda(lambda x: x[:, :, 84:])(inputs)
    x = L.Concatenate(axis=-1)([x, other])
    x = L.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
    x = L.Dense(head_units, activation='relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.5)(x)
    return tf.keras.Model(inputs, L.Dense(11, activation='softmax')(x))


def parameter_counts(model):
    count = lambda weights: int(sum(np.prod(w.shape) for w in weights))
    return {'total': model.count_params(), 'trainable': count(model.trainable_weights), 'non_trainable': count(model.non_trainable_weights)}


def load_data():
    arrays, rows, manifest = [], [], []
    for digit in range(10, 21):
        for path in sorted((SOURCE / 'dataset').glob(f'seq_{digit}_*.npy')):
            a = np.load(path, allow_pickle=False)
            assert a.ndim == 3 and a.shape[1:] == (30, 213), path
            assert np.isfinite(a).all() and np.all(a[:, :, -1] == digit - 10), path
            manifest.append({'file': path.name, 'sequences': len(a), 'sha256': digest(path)})
            arrays.append(a.astype(np.float32))
            rows.extend({'sample_id': f'{path.name}:{i}', 'file': path.name, 'file_index': i, 'label': digit - 10, 'digit': digit} for i in range(len(a)))
    data = np.concatenate(arrays)
    assert len(data) == 976
    x, y = data[:, :, :-1], data[:, 0, -1].astype(np.int64)
    trainval, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
    train, val = train_test_split(trainval, test_size=.125, random_state=42, stratify=y[trainval])
    assert (len(train), len(val), len(test)) == (682, 98, 196)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    for name, ids in [('train', train), ('validation', val), ('test', test)]:
        for i in ids:
            rows[i]['split'] = name
    return x, y, (train, val, test), rows, manifest


def make_dataset(x, y, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((x, tf.keras.utils.to_categorical(y, 11)))
    if seed is not None:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return ds.batch(32).with_options(options).prefetch(1)


def augment(x, seed):
    rng = np.random.RandomState(seed)
    out = x.copy()
    for i in range(len(out)):
        out[i] = out[i] + rng.normal(0, .005, out[i].shape)
        out[i, :, :84] *= rng.uniform(.9, 1.1)
    return out


class Progress(tf.keras.callbacks.Callback):
    def __init__(self, name, seed):
        super().__init__()
        self.name, self.seed, self.start = name, seed, time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        logs['learning_rate'] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        state = {'status': 'training', 'variant': self.name, 'seed': self.seed, 'epoch': epoch + 1, 'elapsed_seconds': round(time.time() - self.start, 2), **logs}
        save_json(OUT / 'status.json', state)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps(state), flush=True)


def run_one(name, seed, x, y, split, rows, protocol_id):
    target = OUT / f'{name}_seed{seed}'
    target.mkdir(exist_ok=True)
    result_file = target / 'result.json'
    if result_file.exists():
        old = json.loads(result_file.read_text(encoding='utf-8'))
        assert old['protocol_id'] == protocol_id, 'Existing result uses a different protocol'
        print('RESUME_COMPLETED', name, seed, flush=True)
        return old
    train, val, test = split
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    dim, head = VARIANTS[name]
    model = build_model(dim, head)
    counts = parameter_counts(model)
    assert counts['total'] == EXPECTED[name], counts
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy'])
    (target / 'model_config.json').write_text(model.to_json(), encoding='utf-8')
    x_train = augment(x[train], seed)[:, :, :dim]
    train_ds = make_dataset(x_train, y[train], seed)
    val_ds = make_dataset(x[val, :, :dim], y[val])
    weights = target / 'best.weights.h5'
    progress = Progress(name, seed)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(weights), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=6, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
        progress,
        tf.keras.callbacks.CSVLogger(str(target / 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    print('START', name, seed, counts, flush=True)
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=callbacks, verbose=0)
    model.load_weights(str(weights))
    h = {k: [float(v) for v in values] for k, values in history.history.items()}
    assert all(np.isfinite(values).all() for values in h.values()), 'Non-finite training history'
    save_json(target / 'history.json', h)
    probability = np.concatenate([model(x[test[i:i+32], :, :dim], training=False).numpy() for i in range(0, len(test), 32)])
    assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1, atol=1e-4)
    predicted, truth = probability.argmax(axis=1), y[test]
    metrics = {'accuracy': float(accuracy_score(truth, predicted)), 'macro_precision': float(precision_score(truth, predicted, average='macro', zero_division=0)), 'macro_recall': float(recall_score(truth, predicted, average='macro', zero_division=0)), 'macro_f1': float(f1_score(truth, predicted, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(truth, predicted, average='weighted', zero_division=0))}
    pred_rows = []
    for j, index in enumerate(test):
        pred_rows.append({'sample_id': rows[index]['sample_id'], 'true_digit': int(truth[j]+10), 'predicted_digit': int(predicted[j]+10), **{f'p_{c+10}': float(probability[j, c]) for c in range(11)}})
    write_csv(target / 'predictions.csv', pred_rows, list(pred_rows[0]))
    cm = confusion_matrix(truth, predicted, labels=list(range(11)))
    assert int(cm.sum()) == 196
    write_csv(target / 'confusion_matrix.csv', [{'true_digit': i+10, **{f'pred_{j+10}': int(cm[i,j]) for j in range(11)}} for i in range(11)], ['true_digit']+[f'pred_{j+10}' for j in range(11)])
    save_json(target / 'classification_report.json', classification_report(truth, predicted, labels=list(range(11)), target_names=[str(i) for i in range(10,21)], output_dict=True, zero_division=0))
    result = {'protocol_id': protocol_id, 'variant': name, 'seed': seed, 'input_dim': dim, 'head_units': head, 'parameters': counts, 'epochs_ran': len(h['loss']), 'best_epoch': int(np.argmin(h['val_loss'])+1), 'minimum_validation_loss': min(h['val_loss']), 'test_samples': len(test), 'seconds': time.time()-start, **metrics}
    save_json(result_file, result)
    print('DONE', json.dumps(result), flush=True)
    return result


def summarize(results):
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']
    summary, csvrows = {}, []
    for name in VARIANTS:
        runs = [r for r in results if r['variant'] == name]
        assert len(runs) == 3 and {r['seed'] for r in runs} == set(SEEDS)
        stats = {k: {'mean': float(np.mean([r[k] for r in runs])), 'sample_std': float(np.std([r[k] for r in runs], ddof=1))} for k in metrics}
        summary[name] = {'runs': 3, 'parameters': runs[0]['parameters'], 'metrics': stats}
        csvrows.append({'variant': name, 'parameters': runs[0]['parameters']['total'], **{f'{k}_{s}': stats[k][s] for k in metrics for s in ['mean','sample_std']}})
    save_json(OUT / 'summary.json', summary)
    write_csv(OUT / 'summary.csv', csvrows, list(csvrows[0]))
    paired = []
    for seed in SEEDS:
        a = {r['variant']: r for r in results if r['seed'] == seed}
        paired.append({'seed': seed, 'full_minus_matched_accuracy_pp': 100*(a['full_delta']['accuracy']-a['no_delta_matched']['accuracy']), 'matched_minus_original_accuracy_pp': 100*(a['no_delta_matched']['accuracy']-a['no_delta_original']['accuracy'])})
    write_csv(OUT / 'paired_differences.csv', paired, list(paired[0]))
    text = ['# Delta容量匹配补充实验结果', '', '每组3次从头训练；表中为均值±样本标准差，单位为百分比（差值的单位为百分点）。', '', '| 模型 | 总参数 | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
    for name, s in summary.items():
        acc, f1 = s['metrics']['accuracy'], s['metrics']['macro_f1']
        text.append(f"| {name} | {s['parameters']['total']:,} | {100*acc['mean']:.2f} ± {100*acc['sample_std']:.2f} | {100*f1['mean']:.2f} ± {100*f1['sample_std']:.2f} |")
    deltas = [r['full_minus_matched_accuracy_pp'] for r in paired]
    text += ['', f'完整模型减容量匹配无Delta模型的准确率差值为 {np.mean(deltas):.2f} ± {np.std(deltas,ddof=1):.2f} 个百分点。', '', '本实验采用现有序列级随机划分，不能证明原视频或参与者独立。3次运行的差值作描述性比较，不自动认定统计显著性。参数总量接近不表示容量在不同层之间的分配完全一致。该实验在新记录的环境中执行，不将其结果冒充旧表格的原始训练记录。', '', '所有逐次配置、训练曲线、权重、预测和混淆矩阵见各运行子目录；协议与数据摘要见protocol.json。']
    (OUT / 'results.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    x, y, split, rows, manifest = load_data()
    design = {'source_data': str(SOURCE / 'dataset'), 'data_manifest': manifest, 'script_sha256': digest(__file__), 'original_ablation_sha256': digest(WORKSPACE/'scripts/core/generate_ablation.py'), 'split_seed': 42, 'split_counts': [len(i) for i in split], 'run_seeds': SEEDS, 'variants': VARIANTS, 'max_epochs': 200, 'batch_size': 32, 'adam_lr': .001, 'lr_factor': .5, 'lr_patience': 6, 'early_stopping_patience': 20, 'selection': 'minimum_validation_loss', 'augmentation_noise_sd': .005, 'augmentation_scale': [.9,1.1], 'standard_deviation_ddof': 1, 'environment': {'python': sys.version, 'tensorflow': tf.__version__, 'keras': keras.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(), 'processor': platform.processor(), 'physical_devices': [str(v) for v in tf.config.list_physical_devices()]}}
    protocol_id = hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    protocol = {'protocol_id': protocol_id, **design}
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))['protocol_id'] == protocol_id, 'Protocol changed; use a separate output directory'
    save_json(OUT / 'protocol.json', protocol)
    write_csv(OUT / 'source_data_manifest.csv', manifest, ['file','sequences','sha256'])
    write_csv(OUT / 'split_manifest.csv', rows, ['sample_id','file','file_index','label','digit','split'])
    checks = {}
    for name, (dim, head) in VARIANTS.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(2021)
        model = build_model(dim,head)
        checks[name] = parameter_counts(model)
        assert checks[name]['total'] == EXPECTED[name]
        if args.preflight:
            model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='categorical_crossentropy')
            loss = model.train_on_batch(x[split[0][:8],:,:dim],tf.keras.utils.to_categorical(y[split[0][:8]],11))
            assert np.isfinite(loss).all()
        print('PARAMETERS',name,checks[name],flush=True)
    assert abs(checks['no_delta_matched']['total']/checks['full_delta']['total']-1)<.01
    assert abs(checks['no_delta_matched']['trainable']/checks['full_delta']['trainable']-1)<.01
    save_json(OUT / 'parameter_check.json',checks)
    if args.preflight:
        print('PREFLIGHT_PASSED_NO_TEST_EVALUATION',flush=True)
        return
    results = []
    try:
        for seed in SEEDS:
            for name in VARIANTS:
                results.append(run_one(name,seed,x,y,split,rows,protocol_id))
        summarize(results)
        save_json(OUT / 'status.json', {'status':'completed','runs_completed':len(results),'protocol_id':protocol_id})
        print('EXPERIMENT_COMPLETED',str(OUT/'results.md'),flush=True)
    except BaseException as e:
        save_json(OUT / 'status.json',{'status':'interrupted_or_failed','error_type':type(e).__name__,'message':str(e),'completed_this_invocation':len(results),'protocol_id':protocol_id})
        raise


if __name__ == '__main__':
    main()
