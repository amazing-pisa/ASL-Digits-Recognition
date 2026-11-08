"""Paired-seed dynamic architecture comparison; CPU, isolated outputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE / 'scripts/core'))
import runtime
import compat  # Applies the tested Windows Keras temporary-directory fix.
from model_builders import build_model
spec = importlib.util.spec_from_file_location('capacity_helpers', HERE.parent / 'delta_capacity/run_experiment.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
np, tf = helpers.np, helpers.tf
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

MODELS = ['stcnn', 'glt', 'lite']
SEEDS = [2021, 2022, 2023]
EXPECTED = dict(stcnn=275403, glt=5712139, lite=119307)
OUT = HERE / 'outputs'
save_json, write_csv, digest = helpers.save_json, helpers.write_csv, helpers.digest


def run_one(kind, seed, x, y, split, rows, protocol_id, epochs):
    target = OUT / f'{kind}_seed{seed}'
    target.mkdir(exist_ok=True)
    result_path = target / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['protocol_id'] == protocol_id, 'Different protocol in existing results'
        assert all((target / p).is_file() for p in ['best.weights.h5', 'predictions.csv', 'history.csv', 'confusion_matrix.csv'])
        print('SKIP_COMPLETED', kind, seed, flush=True)
        return result
    train, val, test = split
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(kind)
    assert model.count_params() == EXPECTED[kind]
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='categorical_crossentropy', metrics=['accuracy'])
    (target / 'model_config.json').write_text(model.to_json(), encoding='utf-8')
    train_ds = helpers.make_dataset(helpers.augment(x[train], seed), y[train], seed)
    val_ds = helpers.make_dataset(x[val], y[val])
    weights = target / 'best.weights.h5'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(weights), monitor='val_loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        helpers.Progress(kind, seed),
        tf.keras.callbacks.CSVLogger(str(target / 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    print('START', kind, seed, flush=True)
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)
    h = {key: [float(v) for v in values] for key, values in history.history.items()}
    assert all(np.isfinite(values).all() for values in h.values())
    save_json(target / 'history.json', h)
    model.load_weights(weights)
    probability = np.concatenate([model(x[test[i:i+32]], training=False).numpy() for i in range(0, len(test), 32)])
    assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1, atol=1e-4)
    truth, predicted = y[test], probability.argmax(axis=1)
    records = [dict(sample_id=rows[index]['sample_id'], true_digit=int(truth[j]+10), predicted_digit=int(predicted[j]+10), **{f'p_{c+10}': float(probability[j,c]) for c in range(11)}) for j, index in enumerate(test)]
    write_csv(target / 'predictions.csv', records, list(records[0]))
    cm = confusion_matrix(truth, predicted, labels=list(range(11)))
    write_csv(target / 'confusion_matrix.csv', [dict(true_digit=i+10, **{f'pred_{j+10}': int(cm[i,j]) for j in range(11)}) for i in range(11)], ['true_digit']+[f'pred_{j+10}' for j in range(11)])
    save_json(target / 'classification_report.json', classification_report(truth, predicted, labels=list(range(11)), output_dict=True, zero_division=0))
    # Verify the saved weights reproduce the predictions before marking complete.
    restored = build_model(kind)
    restored.load_weights(weights)
    np.testing.assert_allclose(restored(x[test[:4]], training=False).numpy(), probability[:4], atol=1e-5, rtol=1e-5)
    result = dict(protocol_id=protocol_id, model=kind, seed=seed, parameters=helpers.parameter_counts(model), epochs_ran=len(h['loss']), best_epoch=int(np.argmin(h['val_loss'])+1), test_samples=len(test), seconds=time.time()-start, accuracy=float(accuracy_score(truth,predicted)), macro_f1=float(f1_score(truth,predicted,average='macro',zero_division=0)))
    save_json(result_path, result)
    print('DONE', kind, seed, 'test_accuracy=', result['accuracy'], flush=True)
    return result


def describe(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = float(stats.t.ppf(.975, n-1)*sd/math.sqrt(n))
    return dict(n=n, mean=mean, sample_std=sd, ci95_low=mean-half, ci95_high=mean+half)


def compare_groups(groups, metrics):
    """Exact two-sided paired sign test; Holm within each family of comparisons."""
    output = []
    for metric in metrics:
        family = []
        for a, b in itertools.combinations(groups, 2):
            assert set(groups[a]) == set(groups[b])
            differences = [groups[a][s][metric]-groups[b][s][metric] for s in sorted(groups[a])]
            positive = sum(v > 1e-12 for v in differences)
            negative = sum(v < -1e-12 for v in differences)
            nonzero = positive+negative
            p = float(stats.binomtest(positive, nonzero, .5).pvalue) if nonzero else 1.
            family.append(dict(metric=metric, model_a=a, model_b=b, differences=differences, difference_stats=describe(differences), nonzero_pairs=nonzero, p_exact_sign=p))
        previous = 0.
        for i, row in enumerate(sorted(family,key=lambda r:r['p_exact_sign'])):
            previous = max(previous, min(1.,(len(family)-i)*row['p_exact_sign']))
            row['p_holm'] = previous
        output.extend(family)
    return output


def summarize(results):
    groups = {kind: {r['seed']:r for r in results if r['model']==kind} for kind in MODELS}
    assert all(set(group)==set(SEEDS) for group in groups.values())
    summary = {kind: {metric: describe([group[s][metric] for s in SEEDS]) for metric in ['accuracy','macro_f1']} for kind,group in groups.items()}
    save_json(OUT / 'summary.json', summary)
    table = [dict(model=kind, metric=metric, **value) for kind,values in summary.items() for metric,value in values.items()]
    write_csv(OUT / 'summary.csv',table,list(table[0]))
    save_json(OUT / 'statistical_tests.json', compare_groups(groups,['accuracy','macro_f1']))
    # Analyze the already completed ablation separately, never pooling test samples across seeds.
    ablation = {}
    for name in ['full_delta','no_delta_original','no_delta_matched']:
        ablation[name] = {}
        for seed in SEEDS:
            path = HERE.parent / 'delta_capacity/outputs' / f'{name}_seed{seed}/result.json'
            if path.exists():
                ablation[name][seed] = json.loads(path.read_text(encoding='utf-8'))
    if all(set(group)==set(SEEDS) for group in ablation.values()):
        assert len({r['protocol_id'] for group in ablation.values() for r in group.values()})==1
        save_json(OUT / 'ablation_statistical_tests.json', compare_groups(ablation,['accuracy','macro_f1']))
    lines = ['# 三种动态架构重复实验', '', '每组3次，表中单位为%；标准差为样本标准差。', '', '| 模型 | Accuracy 均值±SD | Macro-F1 均值±SD |', '|---|---:|---:|']
    for kind in MODELS:
        a,f = summary[kind]['accuracy'],summary[kind]['macro_f1']
        lines.append(f"| {kind} | {100*a['mean']:.2f} ± {100*a['sample_std']:.2f} | {100*f['mean']:.2f} ± {100*f['sample_std']:.2f} |")
    lines += ['', 'summary.csv中的95% t区间描述固定数据划分下不同随机种子的均值不确定性；仅3次运行，正态假设难以检查，区间可能超出指标范围，不代表参与者或视频总体的不确定性。', '', 'statistical_tests.json采用双侧精确配对符号检验，对每个指标的3项架构比较进行Holm校正。该检验考察差异方向，不直接检验均值相等。3个非零配对的最小双侧p值为0.25，无法在0.05水平给出显著结论；不显著不代表模型等效。', '', '三次测试使用相同196条序列，不能作为588条独立测试样本。现有数据缺少参与者/原视频/录制时段标识，无法保证相邻或相似原始片段完全隔离；本实验仅支持序列级划分结论。统一训练预算不等于已完成各模型独立超参数搜索。', '', '已完成的容量消融统计另存ablation_statistical_tests.json；该文件仅在9份旧结果均存在且协议一致时生成。新结果不能冒充原论文表格的历史记录。']
    (OUT / 'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='One epoch per model/seed in separate smoke_outputs; not paper results')
    args = parser.parse_args()
    if args.check:
        OUT = HERE / 'smoke_outputs'
    OUT.mkdir(exist_ok=True)
    helpers.OUT = OUT
    helpers.SOURCE = runtime.SOURCE
    tf.config.experimental.enable_op_determinism()
    x,y,split,rows,manifest = helpers.load_data()
    design = dict(script_sha256=digest(__file__), helper_sha256=digest(helpers.__file__), builders_sha256=digest(WORKSPACE/'models/model_builders.py'), runtime_sha256=digest(runtime.__file__), compat_sha256=digest(compat.__file__), source_manifest=manifest, split_seed=42, split_counts=[len(i) for i in split], models=MODELS, seeds=SEEDS, max_epochs=1 if args.check else 200, batch_size=32, adam_lr=.001, lr_factor=.5, lr_patience=6, early_patience=20, selection='min_val_loss', train_only_augmentation=dict(noise_sd=.005,scale=[.9,1.1],scaled_features='first84'), environment=dict(python=sys.version,tensorflow=tf.__version__,keras=helpers.keras.__version__,numpy=np.__version__,scipy=__import__('scipy').__version__,sklearn=helpers.sklearn.__version__,platform=platform.platform(),devices=[str(v) for v in tf.config.list_physical_devices()]), smoke_check=args.check)
    protocol_id = hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    protocol_path = OUT/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8'))['protocol_id']==protocol_id, 'Protocol changed. Preserve old outputs and use a separate output folder.'
    save_json(protocol_path,dict(protocol_id=protocol_id,**design))
    write_csv(OUT/'source_data_manifest.csv',manifest,list(manifest[0]))
    write_csv(OUT/'split_manifest.csv',rows,list(rows[0]))
    # Prevent simultaneous runs writing the same checkpoints.
    lock = OUT/'RUNNING.lock'
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise RuntimeError('An experiment may be running. If no training window remains, remove RUNNING.lock then retry.')
    try:
        results = []
        for seed in SEEDS:
            for kind in MODELS:
                results.append(run_one(kind,seed,x,y,split,rows,protocol_id,design['max_epochs']))
        if not args.check:
            summarize(results)
        save_json(OUT/'status.json',dict(status='completed',runs=len(results),smoke_check=args.check))
        print('CHECK_PASSED' if args.check else 'ALL_TRAINING_COMPLETED', str(OUT),flush=True)
    except BaseException as e:
        save_json(OUT/'status.json',dict(status='interrupted_or_failed',error=str(e)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
