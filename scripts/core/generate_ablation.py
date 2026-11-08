import runtime
from compat import load_model_compat, read_image, normalize_fnn_batch, standardize_vgg, save_scaler, initialize_vgg_backbone
"""
消融实验脚本
功能：分别对1D-CNN和ST-CNN进行消融实验，训练各变体模型，评估并生成对比图表
运行方式：python generate_ablation.py
输出目录：paper/ablation/
"""

import os
import json
import glob
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Conv1D, MaxPooling1D, Flatten,
    Dropout, BatchNormalization, Conv2D, Reshape, Concatenate, Lambda,
    GlobalAveragePooling1D, GlobalMaxPooling1D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===========================
# 0. 全局配置
# ===========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'paper', 'ablation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_DATASET = str(runtime.STATIC_DATASET)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===========================
# 1. 静态数据加载
# ===========================
print("=" * 60)
print("  初始化环境...")
print("=" * 60)

mp_hands = mp.solutions.hands
hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def extract_landmarks_3d(image_path):
    image = read_image(image_path)
    if image is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_static.process(image_rgb)
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            lm = np.array([[l.x, l.y, l.z] for l in hl.landmark])
            wrist = lm[0]
            lm = lm - wrist
            d = np.max(np.linalg.norm(lm, axis=1))
            if d > 0: lm = lm / d
            return lm  # shape (21, 3)
    return None


def load_static_dataset(split="train"):
    X, y = [], []
    split_path = os.path.join(STATIC_DATASET, split)
    if not os.path.exists(split_path):
        return np.array([]), np.array([])
    for label_folder in sorted(os.listdir(split_path)):
        label_path = os.path.join(split_path, label_folder)
        if not os.path.isdir(label_path): continue
        try:
            label = int(label_folder.replace("A", ""))
        except ValueError:
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            lm = extract_landmarks_3d(img_path)
            if lm is not None:
                X.append(lm)
                y.append(label)
    return np.array(X), np.array(y)


# ===========================
# 2. 动态数据加载
# ===========================
ACTIONS_DYN = [str(i) for i in range(10, 21)]


def load_dynamic_dataset():
    data_list = []
    dataset_dir = str(runtime.DYNAMIC_DATASET)
    for action in ACTIONS_DYN:
        matched = glob.glob(os.path.join(dataset_dir, f'seq_{action}_*.npy'))
        for f in matched:
            loaded = np.load(f)
            if loaded.ndim == 3:
                data_list.append(loaded)
    data = np.concatenate(data_list, axis=0)
    x_data = data[:, :, :-1].astype(np.float32)
    labels = data[:, 0, -1]
    y_data = to_categorical(labels, num_classes=len(ACTIONS_DYN)).astype(np.float32)
    return x_data, y_data


# ===========================
# 3. 1D-CNN 消融模型定义
# ===========================
def build_cnn_full(input_shape=(21, 3)):
    """完整1D-CNN（基线）"""
    return Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])


def build_cnn_no_bn(input_shape=(21, 3)):
    """去除BatchNormalization"""
    return Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(256, 3, activation='relu', padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])


def build_cnn_no_dropout(input_shape=(21, 3)):
    """去除Dropout"""
    return Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])


def build_cnn_2layers(input_shape=(21, 3)):
    """减少至2层卷积"""
    return Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])


# ===========================
# 4. ST-CNN 消融模型定义
# ===========================
NUM_DYN_CLASSES = len(ACTIONS_DYN)


def build_stcnn_full(input_dim=212):
    """完整ST-CNN（基线）"""
    inputs = Input(shape=(30, input_dim))
    joints_raw = Lambda(lambda x: x[:, :, :84])(inputs)
    joints_spatial = Reshape((30, 21, 4))(joints_raw)
    x_s = Conv2D(64, (1, 3), activation='relu', padding='same')(joints_spatial)
    x_s = BatchNormalization()(x_s)
    x_t = Conv2D(128, (3, 1), activation='relu', padding='same')(x_s)
    x_t = BatchNormalization()(x_t)
    x_seq = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x_t)
    other = Lambda(lambda x: x[:, :, 84:])(inputs)
    concat = Concatenate(axis=-1)([x_seq, other])
    tcn = Conv1D(128, 5, activation='relu', padding='same')(concat)
    tcn = BatchNormalization()(tcn)
    tcn = Conv1D(128, 3, activation='relu', padding='same')(tcn)
    tcn = BatchNormalization()(tcn)
    avg_p = GlobalAveragePooling1D()(tcn)
    max_p = GlobalMaxPooling1D()(tcn)
    feat = Concatenate()([avg_p, max_p])
    z = Dense(128, activation='relu')(feat)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    outputs = Dense(NUM_DYN_CLASSES, activation='softmax')(z)
    return Model(inputs, outputs)


def build_stcnn_no_spatial(input_dim=212):
    """去除空间卷积（Conv2D → Dense投影）"""
    inputs = Input(shape=(30, input_dim))
    joints_raw = Lambda(lambda x: x[:, :, :84])(inputs)
    x_proj = Dense(128, activation='relu')(joints_raw)
    other = Lambda(lambda x: x[:, :, 84:])(inputs)
    concat = Concatenate(axis=-1)([x_proj, other])
    tcn = Conv1D(128, 5, activation='relu', padding='same')(concat)
    tcn = BatchNormalization()(tcn)
    tcn = Conv1D(128, 3, activation='relu', padding='same')(tcn)
    tcn = BatchNormalization()(tcn)
    avg_p = GlobalAveragePooling1D()(tcn)
    max_p = GlobalMaxPooling1D()(tcn)
    feat = Concatenate()([avg_p, max_p])
    z = Dense(128, activation='relu')(feat)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    outputs = Dense(NUM_DYN_CLASSES, activation='softmax')(z)
    return Model(inputs, outputs)


def build_stcnn_no_tcn(input_dim=212):
    """去除时间卷积（TCN → 直接池化）"""
    inputs = Input(shape=(30, input_dim))
    joints_raw = Lambda(lambda x: x[:, :, :84])(inputs)
    joints_spatial = Reshape((30, 21, 4))(joints_raw)
    x_s = Conv2D(64, (1, 3), activation='relu', padding='same')(joints_spatial)
    x_s = BatchNormalization()(x_s)
    x_t = Conv2D(128, (3, 1), activation='relu', padding='same')(x_s)
    x_t = BatchNormalization()(x_t)
    x_seq = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x_t)
    other = Lambda(lambda x: x[:, :, 84:])(inputs)
    concat = Concatenate(axis=-1)([x_seq, other])
    avg_p = GlobalAveragePooling1D()(concat)
    max_p = GlobalMaxPooling1D()(concat)
    feat = Concatenate()([avg_p, max_p])
    z = Dense(128, activation='relu')(feat)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    outputs = Dense(NUM_DYN_CLASSES, activation='softmax')(z)
    return Model(inputs, outputs)


def build_stcnn_no_mixpool(input_dim=212):
    """去除混合池化（仅GAP）"""
    inputs = Input(shape=(30, input_dim))
    joints_raw = Lambda(lambda x: x[:, :, :84])(inputs)
    joints_spatial = Reshape((30, 21, 4))(joints_raw)
    x_s = Conv2D(64, (1, 3), activation='relu', padding='same')(joints_spatial)
    x_s = BatchNormalization()(x_s)
    x_t = Conv2D(128, (3, 1), activation='relu', padding='same')(x_s)
    x_t = BatchNormalization()(x_t)
    x_seq = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x_t)
    other = Lambda(lambda x: x[:, :, 84:])(inputs)
    concat = Concatenate(axis=-1)([x_seq, other])
    tcn = Conv1D(128, 5, activation='relu', padding='same')(concat)
    tcn = BatchNormalization()(tcn)
    tcn = Conv1D(128, 3, activation='relu', padding='same')(tcn)
    tcn = BatchNormalization()(tcn)
    feat = GlobalAveragePooling1D()(tcn)
    z = Dense(128, activation='relu')(feat)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    outputs = Dense(NUM_DYN_CLASSES, activation='softmax')(z)
    return Model(inputs, outputs)


def build_stcnn_no_delta(input_dim=106):
    """去除Delta速度特征（输入106维）"""
    inputs = Input(shape=(30, input_dim))
    joints_raw = Lambda(lambda x: x[:, :, :84])(inputs)
    joints_spatial = Reshape((30, 21, 4))(joints_raw)
    x_s = Conv2D(64, (1, 3), activation='relu', padding='same')(joints_spatial)
    x_s = BatchNormalization()(x_s)
    x_t = Conv2D(128, (3, 1), activation='relu', padding='same')(x_s)
    x_t = BatchNormalization()(x_t)
    x_seq = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x_t)
    other = Lambda(lambda x: x[:, :, 84:])(inputs)
    concat = Concatenate(axis=-1)([x_seq, other])
    tcn = Conv1D(128, 5, activation='relu', padding='same')(concat)
    tcn = BatchNormalization()(tcn)
    tcn = Conv1D(128, 3, activation='relu', padding='same')(tcn)
    tcn = BatchNormalization()(tcn)
    avg_p = GlobalAveragePooling1D()(tcn)
    max_p = GlobalMaxPooling1D()(tcn)
    feat = Concatenate()([avg_p, max_p])
    z = Dense(128, activation='relu')(feat)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    outputs = Dense(NUM_DYN_CLASSES, activation='softmax')(z)
    return Model(inputs, outputs)


# ===========================
# 5. 训练与评估工具
# ===========================
def train_and_eval(model, x_train, y_train, x_val, y_val, x_test, y_test,
                   epochs=50, batch_size=32, patience=12):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(4, patience // 3), verbose=0),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
    ]
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    params = model.count_params()
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'params': params}


# ===========================
# 6. 图表生成
# ===========================
def plot_ablation_bar(results, title, save_path, metrics_keys=None):
    """生成消融实验柱状图"""
    if metrics_keys is None:
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    labels_cn = {'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率', 'f1': 'F1值'}

    names = list(results.keys())
    n = len(names)
    x = np.arange(len(metrics_keys))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    for i, name in enumerate(names):
        vals = [results[name][k] * 100 for k in metrics_keys]
        bars = ax.bar(x + i * width - (n - 1) * width / 2, vals, width,
                      label=name, color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('百分比 (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([labels_cn.get(k, k) for k in metrics_keys], fontsize=11)
    ax.set_ylim(bottom=max(0, min(
        min(results[n][k] * 100 for k in metrics_keys) for n in names) - 10), top=105)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存: {save_path}")


def plot_ablation_delta(results, baseline_key, title, save_path):
    """生成消融实验精度下降对比图（相对于基线的Accuracy差异）"""
    baseline_acc = results[baseline_key]['accuracy'] * 100
    names = [k for k in results if k != baseline_key]
    deltas = [results[k]['accuracy'] * 100 - baseline_acc for k in names]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas]
    bars = ax.barh(range(len(names)), deltas, color=colors, edgecolor='white', height=0.6)

    for bar, delta in zip(bars, deltas):
        xpos = bar.get_width() + (0.15 if delta >= 0 else -0.15)
        ha = 'left' if delta >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{delta:+.2f}%', ha=ha, va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('准确率变化 (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存: {save_path}")


def generate_md_table(results, title):
    """输出Markdown表格"""
    print(f"\n===== {title} =====\n")
    print("| 模型变体 | Accuracy | Precision | Recall | F1(Macro) | 参数量 |")
    print("|---------|----------|-----------|--------|-----------|--------|")
    for name, m in results.items():
        print(f"| {name} | {m['accuracy']*100:.2f}% | {m['precision']*100:.2f}% | "
              f"{m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {m['params']:,} |")


# ===========================
# 7. 运行1D-CNN消融实验
# ===========================
def run_static_ablation():
    print("\n" + "=" * 60)
    print("  1D-CNN 消融实验（静态手势 0-9）")
    print("=" * 60)

    print("\n[1/5] 加载静态手势数据集...")
    X_train_raw, y_train_raw = load_static_dataset("train")
    X_val_raw, y_val_raw = load_static_dataset("valid")
    X_test_raw, y_test_raw = load_static_dataset("test")
    print(f"  训练集: {len(X_train_raw)}, 验证集: {len(X_val_raw)}, 测试集: {len(X_test_raw)}")

    # 数据增强（高斯噪声）
    def augment(X):
        return X + np.random.normal(0, 0.005, X.shape)

    # 预处理: 3D 版本
    X_train_3d = augment(X_train_raw.copy()).reshape(-1, 21, 3)
    X_val_3d = X_val_raw.reshape(-1, 21, 3)
    X_test_3d = X_test_raw.reshape(-1, 21, 3)

    # 预处理: 2D 版本（去掉z轴）
    X_train_2d = augment(X_train_raw.copy()).reshape(-1, 21, 3)[:, :, :2]
    X_val_2d = X_val_raw.reshape(-1, 21, 3)[:, :, :2]
    X_test_2d = X_test_raw.reshape(-1, 21, 3)[:, :, :2]

    # 无增强版本
    X_train_noaug = X_train_raw.copy().reshape(-1, 21, 3)

    y_train = to_categorical(y_train_raw, 10)
    y_val = to_categorical(y_val_raw, 10)
    y_test = to_categorical(y_test_raw, 10)

    results = {}
    N_RUNS = 3  # 每个变体多次运行取平均

    ablation_configs = [
        ("完整模型(基线)", lambda: build_cnn_full((21, 3)), X_train_3d, X_val_3d, X_test_3d),
        ("去除深度信息(2D)", lambda: build_cnn_full((21, 2)), X_train_2d, X_val_2d, X_test_2d),
        ("去除批归一化", lambda: build_cnn_no_bn((21, 3)), X_train_3d, X_val_3d, X_test_3d),
        ("去除Dropout", lambda: build_cnn_no_dropout((21, 3)), X_train_3d, X_val_3d, X_test_3d),
        ("减少卷积层(2层)", lambda: build_cnn_2layers((21, 3)), X_train_3d, X_val_3d, X_test_3d),
    ]

    for idx, (name, build_fn, xt, xv, xte) in enumerate(ablation_configs):
        print(f"\n[{idx+2}/5] 训练消融变体: {name}...")
        run_results = []
        for r in range(N_RUNS):
            tf.keras.backend.clear_session()
            model = build_fn()
            res = train_and_eval(model, xt, y_train, xv, y_val, xte, y_test,
                                 epochs=50, batch_size=32, patience=12)
            run_results.append(res)
            print(f"    Run {r+1}/{N_RUNS}: Acc={res['accuracy']*100:.2f}%")
        # 取平均
        avg_res = {}
        for key in run_results[0]:
            if key == 'params':
                avg_res[key] = run_results[0][key]
            else:
                avg_res[key] = np.mean([r[key] for r in run_results])
        results[name] = avg_res
        print(f"  => 平均 Accuracy: {avg_res['accuracy']*100:.2f}%, F1: {avg_res['f1']*100:.2f}%")

    # 生成图表
    print("\n生成1D-CNN消融实验图表...")
    plot_ablation_bar(results, '1D-CNN 消融实验对比',
                      os.path.join(OUTPUT_DIR, 'static_ablation_bar.png'))
    plot_ablation_delta(results, '完整模型(基线)', '1D-CNN 消融实验 — 各组件移除后准确率变化',
                        os.path.join(OUTPUT_DIR, 'static_ablation_delta.png'))
    generate_md_table(results, '1D-CNN 消融实验结果 (Markdown)')

    # 保存JSON
    json_path = os.path.join(OUTPUT_DIR, 'static_ablation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> 已保存: {json_path}")

    return results


# ===========================
# 8. 运行ST-CNN消融实验
# ===========================
def run_dynamic_ablation():
    print("\n" + "=" * 60)
    print("  ST-CNN 消融实验（动态手势 10-20）")
    print("=" * 60)

    print("\n[1/6] 加载动态手势数据集...")
    x_all, y_all = load_dynamic_dataset()
    print(f"  总样本数: {len(x_all)}")

    # 划分训练/验证/测试集
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=SEED, stratify=np.argmax(y_all, axis=1))
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.125, random_state=SEED,
        stratify=np.argmax(y_trainval, axis=1))
    print(f"  训练集: {len(x_train)}, 验证集: {len(x_val)}, 测试集: {len(x_test)}")

    # 数据增强
    def augment_dyn(x):
        x_aug = x.copy()
        noise = np.random.normal(0, 0.005, x_aug.shape)
        x_aug = x_aug + noise
        scale = np.random.uniform(0.9, 1.1)
        x_aug[:, :84] = x_aug[:, :84] * scale
        return x_aug

    x_train_aug = np.array([augment_dyn(s) for s in x_train])

    # 仅前106维（去除Delta特征）
    x_train_106 = x_train_aug[:, :, :106]
    x_val_106 = x_val[:, :, :106]
    x_test_106 = x_test[:, :, :106]

    results = {}
    N_RUNS = 3

    ablation_configs = [
        ("完整模型(基线)", lambda: build_stcnn_full(212), x_train_aug, x_val, x_test),
        ("去除空间卷积", lambda: build_stcnn_no_spatial(212), x_train_aug, x_val, x_test),
        ("去除Delta特征", lambda: build_stcnn_no_delta(106), x_train_106, x_val_106, x_test_106),
    ]

    for idx, (name, build_fn, xt, xv, xte) in enumerate(ablation_configs):
        print(f"\n[{idx+2}/6] 训练消融变体: {name}...")
        run_results = []
        for r in range(N_RUNS):
            tf.keras.backend.clear_session()
            model = build_fn()
            res = train_and_eval(model, xt, y_train, xv, y_val, xte, y_test,
                                 epochs=200, batch_size=32, patience=20)
            run_results.append(res)
            print(f"    Run {r+1}/{N_RUNS}: Acc={res['accuracy']*100:.2f}%")
        avg_res = {}
        for key in run_results[0]:
            if key == 'params':
                avg_res[key] = run_results[0][key]
            else:
                avg_res[key] = np.mean([r[key] for r in run_results])
        results[name] = avg_res
        print(f"  => 平均 Accuracy: {avg_res['accuracy']*100:.2f}%, F1: {avg_res['f1']*100:.2f}%")

    # 生成图表
    print("\n生成ST-CNN消融实验图表...")
    plot_ablation_bar(results, 'ST-CNN 消融实验对比',
                      os.path.join(OUTPUT_DIR, 'dynamic_ablation_bar.png'))
    plot_ablation_delta(results, '完整模型(基线)', 'ST-CNN 消融实验 — 各组件移除后准确率变化',
                        os.path.join(OUTPUT_DIR, 'dynamic_ablation_delta.png'))
    generate_md_table(results, 'ST-CNN 消融实验结果 (Markdown)')

    # 保存JSON
    json_path = os.path.join(OUTPUT_DIR, 'dynamic_ablation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> 已保存: {json_path}")

    return results


# ===========================
# 9. 主函数
# ===========================
if __name__ == '__main__':
    static_res = run_static_ablation()
    dynamic_res = run_dynamic_ablation()

    print("\n" + "=" * 60)
    print("  全部消融实验完成！")
    print(f"  所有图表和数据已保存至: {OUTPUT_DIR}")
    print("=" * 60)
