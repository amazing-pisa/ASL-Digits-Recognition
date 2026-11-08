import runtime
from compat import load_model_compat, read_image, normalize_fnn_batch, standardize_vgg, save_scaler, initialize_vgg_backbone
"""
对比实验自动评估脚本
功能：自动计算所有模型指标 + 生成对比图表 + 输出LaTeX/Markdown表格
运行方式：python generate_comparison.py
输出目录：paper/comparison/
"""

import os
import cv2
import time
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, LayerNormalization, MultiHeadAttention
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ===========================
# 0. 全局配置
# ===========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'paper/comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_DATASET = str(runtime.STATIC_DATASET)
DYNAMIC_TESTDIR = runtime.CONFIG.get('dynamic_test_dir') or str(runtime.ROOT/'missing_dynamic_videos')

mp_hands = mp.solutions.hands
hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
hands_video = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

SEQ_LEN = 30
ACTIONS_DYNAMIC = [str(i) for i in range(10, 21)]


# ===========================
# 1. 静态手势特征提取
# ===========================
def extract_landmarks_63(image_path):
    image = read_image(image_path)
    if image is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_static.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            wrist = lm_array[0]
            lm_array = lm_array - wrist
            max_dist = np.max(np.linalg.norm(lm_array, axis=1))
            if max_dist > 0:
                lm_array = lm_array / max_dist
            return lm_array.flatten()
    return None


def load_static_dataset(split="test"):
    X, y, paths = [], [], []
    split_path = os.path.join(STATIC_DATASET, split)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Path not found: {split_path}")
    for label_folder in sorted(os.listdir(split_path)):
        label_path = os.path.join(split_path, label_folder)
        if not os.path.isdir(label_path): continue
        try:
            label = int(label_folder.replace("A", ""))
        except ValueError:
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            landmarks = extract_landmarks_63(img_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
                paths.append(img_path)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


def load_static_dual_dataset(split="test"):
    X_img, X_lm, y = [], [], []
    split_path = os.path.join(STATIC_DATASET, split)
    for label_folder in sorted(os.listdir(split_path)):
        label_path = os.path.join(split_path, label_folder)
        if not os.path.isdir(label_path): continue
        try:
            label = int(label_folder.replace("A", ""))
        except ValueError:
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            image = read_image(img_path)
            if image is None: continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands_static.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    img_resized = cv2.resize(image_rgb, (224, 224))
                    X_img.append(img_resized)
                    X_lm.append(lm_array)
                    y.append(label)
                    break
    return np.array(X_img, dtype=np.float32), np.array(X_lm, dtype=np.float32), np.array(y, dtype=np.int32)


# ===========================
# 2. 动态手势特征提取
# ===========================
def extract_keypoints_from_video(video_path, seq_len=30):
    cap = cv2.VideoCapture(video_path)
    keypoints_seq = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_video.process(img_rgb)
        if not results.multi_hand_landmarks: continue
        res = results.multi_hand_landmarks[0]
        joint = np.zeros((21, 4))
        base = [res.landmark[0].x, res.landmark[0].y, res.landmark[0].z]
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x - base[0], lm.y - base[1], lm.z - base[2], lm.visibility]
        max_dist = np.max(np.linalg.norm(joint[:, :3], axis=1))
        if max_dist == 0: max_dist = 1e-5
        joint[:, :3] /= max_dist
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
        v = (v2 - v1) / (np.linalg.norm(v2 - v1, axis=1)[:, np.newaxis] + 1e-5)
        angle = np.degrees(np.arccos(np.clip(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]), -1.0, 1.0)))
        pinch_20 = np.linalg.norm(joint[8,:3] - joint[4,:3])
        index_to_wrist = np.linalg.norm(joint[8,:3] - joint[0,:3])
        middle_to_wrist = np.linalg.norm(joint[12,:3] - joint[0,:3])
        ring_to_wrist = np.linalg.norm(joint[16,:3] - joint[0,:3])
        pinky_to_wrist = np.linalg.norm(joint[20,:3] - joint[0,:3])
        wrist_twist_z = joint[5,2] - joint[17,2]
        wrist_twist_x = np.linalg.norm(joint[5,:2] - joint[17,:2])
        asl_features = [pinch_20, index_to_wrist, middle_to_wrist, ring_to_wrist, pinky_to_wrist, wrist_twist_z, wrist_twist_x]
        frame_keypoints = np.concatenate([joint.flatten(), angle, asl_features])
        keypoints_seq.append(frame_keypoints)
    cap.release()
    video_length = len(keypoints_seq)
    if video_length == 0: return None
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


def load_dynamic_testset():
    DYNAMIC_TESTDIR = runtime.dynamic_test_dir()
    x_test, y_test = [], []
    for label_str in sorted(os.listdir(DYNAMIC_TESTDIR), key=lambda x: int(x) if x.isdigit() else -1):
        label_dir = os.path.join(DYNAMIC_TESTDIR, label_str)
        if not os.path.isdir(label_dir): continue
        for video_file in os.listdir(label_dir):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')): continue
            seq = extract_keypoints_from_video(os.path.join(label_dir, video_file), SEQ_LEN)
            if seq is not None:
                x_test.append(seq)
                y_test.append(int(label_str) - 10)
    return np.array(x_test), np.array(y_test)


# ===========================
# 3. GLT 模型构建（需要自定义层）
# ===========================
from model_builders import AddCLSToken

from model_builders import PositionalEmbedding


def build_st_cnn_model():
    """重建 ST-CNN 模型结构（避免 Lambda 反序列化问题）"""
    from tensorflow.keras.layers import (Conv2D, Reshape, Concatenate, Conv1D,
        BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda)
    inputs = Input(shape=(30, 212))
    joints_raw = Lambda(lambda x: x[:, :, :84])(inputs)
    joints_spatial = Reshape((30, 21, 4))(joints_raw)
    x_spatial = Conv2D(64, kernel_size=(1, 3), activation='relu', padding='same')(joints_spatial)
    x_spatial = BatchNormalization()(x_spatial)
    x_temporal = Conv2D(128, kernel_size=(3, 1), activation='relu', padding='same')(x_spatial)
    x_temporal = BatchNormalization()(x_temporal)
    x_seq = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x_temporal)
    other_features = Lambda(lambda x: x[:, :, 84:])(inputs)
    concat_seq = Concatenate(axis=-1)([x_seq, other_features])
    tcn_out = Conv1D(128, kernel_size=5, activation='relu', padding='same')(concat_seq)
    tcn_out = BatchNormalization()(tcn_out)
    tcn_out = Conv1D(128, kernel_size=3, activation='relu', padding='same')(tcn_out)
    tcn_out = BatchNormalization()(tcn_out)
    avg_pool = GlobalAveragePooling1D()(tcn_out)
    max_pool = GlobalMaxPooling1D()(tcn_out)
    final_features = Concatenate()([avg_pool, max_pool])
    z = Dense(128, activation='relu')(final_features)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    outputs = Dense(len(ACTIONS_DYNAMIC), activation='softmax')(z)
    model = Model(inputs, outputs, name="Lite_TCN_Optimized")
    return model


def build_transformer_lite_model():
    """重建 Transformer Lite 模型结构"""
    from tensorflow.keras.layers import Add, GlobalAveragePooling1D
    inputs = Input(shape=(30, 212))
    x = Dense(128, activation='relu')(inputs)
    attention_output = MultiHeadAttention(num_heads=2, key_dim=64, dropout=0.1)(x, x)
    res1 = Add()([x, attention_output])
    norm1 = LayerNormalization(epsilon=1e-6)(res1)
    ffn_output = Dense(128, activation="relu")(norm1)
    ffn_output = Dropout(0.1)(ffn_output)
    res2 = Add()([norm1, ffn_output])
    norm2 = LayerNormalization(epsilon=1e-6)(res2)
    out = GlobalAveragePooling1D()(norm2)
    out = Dense(64, activation='relu')(out)
    out = Dropout(0.3)(out)
    outputs = Dense(len(ACTIONS_DYNAMIC), activation='softmax')(out)
    model = Model(inputs, outputs)
    return model


def build_glt_model():
    inputs = Input(shape=(30, 212))
    hidden_dim = 768
    projected_features = Dense(hidden_dim)(inputs)
    glt_input = AddCLSToken(hidden_dim)(projected_features)
    glt_input = PositionalEmbedding(seq_len=31, hidden_dim=hidden_dim)(glt_input)
    mask = np.zeros((31, 31), dtype=bool)
    for i in range(31):
        for j in range(31):
            if i == 0 or j == 0 or abs(i - j) <= 2:
                mask[i, j] = True
    mask_tensor = tf.constant(mask, dtype=tf.bool)
    attn_output = MultiHeadAttention(num_heads=8, key_dim=hidden_dim // 8, dropout=0.1)(
        glt_input, glt_input, attention_mask=mask_tensor)
    glt_out = LayerNormalization(epsilon=1e-6)(glt_input + attn_output)
    ffn_output = Dense(2048, activation='relu')(glt_out)
    ffn_output = Dropout(0.1)(ffn_output)
    ffn_output = Dense(hidden_dim)(ffn_output)
    glt_out = LayerNormalization(epsilon=1e-6)(glt_out + ffn_output)
    cls_output = Lambda(lambda x: x[:, 0, :])(glt_out)
    z = LayerNormalization(epsilon=1e-6)(cls_output)
    outputs = Dense(len(ACTIONS_DYNAMIC), activation='softmax')(z)
    model = Model(inputs, outputs, name="Pure_GLT_Baseline")
    return model


# ===========================
# 4. 评估工具函数
# ===========================
def evaluate_model_metrics(y_true, y_pred, class_names):
    """计算所有指标并返回字典"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'per_class': {}
    }
    for cls in class_names:
        if cls in report:
            metrics['per_class'][cls] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1': report[cls]['f1-score'],
                'support': report[cls]['support']
            }
    return metrics


def measure_inference_time(model, sample_input, n_runs=50):
    """测量单样本推理时间(ms)"""
    # 预热
    for _ in range(5):
        model.predict(sample_input, verbose=0)
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(sample_input, verbose=0)
        times.append((time.time() - start) * 1000)
    return np.mean(times), np.std(times)


def count_parameters(model):
    """统计模型参数量"""
    return model.count_params()


def get_model_size_mb(model_path):
    """获取模型文件大小（MB）"""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0


# ===========================
# 5. 绘图函数
# ===========================
def plot_accuracy_comparison_bar(names, accuracies, title, save_path):
    """总准确率对比柱状图"""
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(names, [a * 100 for a in accuracies], color=colors[:len(names)], width=0.3, edgecolor='black', linewidth=0.8)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=13)
    ax.set_title(title, fontsize=15, pad=15)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存: {save_path}")


def plot_per_class_f1_grouped(all_metrics, model_names, class_names, title, save_path):
    """各类别F1值分组柱状图"""
    n_models = len(model_names)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / n_models
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 1.2), 6), dpi=300)
    for i, (metrics, name) in enumerate(zip(all_metrics, model_names)):
        f1_values = [metrics['per_class'].get(cls, {}).get('f1', 0) * 100 for cls in class_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_values, width, label=name, color=colors[i % len(colors)],
                      edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('手势类别', fontsize=17)
    ax.set_ylabel('F1值 (%)', fontsize=17)
    ax.set_title(title, fontsize=19, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存: {save_path}")


def plot_metrics_comparison_table(all_metrics, model_names, param_counts, infer_times, model_sizes, title, save_path):
    """综合指标对比表（图片版）"""
    headers = ['模型', 'Accuracy', 'Precision', 'Recall', 'F1(Macro)', 'F1(Weighted)', '参数量', '推理时间(ms)', '模型大小(MB)']
    rows = []
    for i, (m, name) in enumerate(zip(all_metrics, model_names)):
        rows.append([
            name,
            f"{m['accuracy']*100:.2f}%",
            f"{m['macro_precision']*100:.2f}%",
            f"{m['macro_recall']*100:.2f}%",
            f"{m['macro_f1']*100:.2f}%",
            f"{m['weighted_f1']*100:.2f}%",
            f"{param_counts[i]:,}",
            f"{infer_times[i]:.1f}",
            f"{model_sizes[i]:.1f}"
        ])

    fig, ax = plt.subplots(figsize=(16, 2 + len(rows) * 0.8), dpi=300)
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2196F3')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5')
        cell.set_edgecolor('#cccccc')

    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存: {save_path}")


def plot_param_infer_comparison(model_names, param_counts, infer_times, title, save_path):
    """参数量与推理时间双轴柱状图"""
    x = np.arange(len(model_names))
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    color1 = '#2196F3'
    bars1 = ax1.bar(x - 0.2, [p / 1000 for p in param_counts], 0.35, label='参数量 (K)',
                    color=color1, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('参数量 (千)', color=color1, fontsize=15)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=13)

    ax2 = ax1.twinx()
    color2 = '#FF5722'
    bars2 = ax2.bar(x + 0.2, infer_times, 0.35, label='推理时间 (ms)',
                    color=color2, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('推理时间 (ms)', color=color2, fontsize=15)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=13)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=14)
    ax1.set_title(title, fontsize=17, pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=13)

    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存: {save_path}")


# ===========================
# 6. 主流程
# ===========================
def run_static_comparison():
    """静态手势（0-10）三模型对比"""
    print("\n" + "="*60)
    print("  静态手势识别（0-10）对比实验")
    print("="*60)

    class_names = [str(i) for i in range(10)]
    results = {}

    # --- 加载测试集 ---
    print("\n[1/6] 加载静态手势测试集...")
    X_test_63, y_test, _ = load_static_dataset("test")

    # --- 模型1: 1D-CNN (本文) ---
    print("\n[2/6] 评估 1D-CNN (本文模型)...")
    cnn_path = runtime.model_path('cnn')
    model_cnn = load_model_compat(cnn_path)
    X_cnn = X_test_63.reshape(-1, 21, 3)
    y_pred_cnn = np.argmax(model_cnn.predict(X_cnn, verbose=0), axis=1)
    metrics_cnn = evaluate_model_metrics(y_test, y_pred_cnn, class_names)
    t_cnn, _ = measure_inference_time(model_cnn, X_cnn[:1])
    params_cnn = count_parameters(model_cnn)
    size_cnn = get_model_size_mb(cnn_path)
    print(f"  Accuracy: {metrics_cnn['accuracy']*100:.2f}%, Params: {params_cnn:,}, Time: {t_cnn:.1f}ms")

    # --- 模型2: FNN ---
    print("\n[3/6] 评估 FNN (对比模型)...")
    fnn_path = runtime.model_path('fnn')
    model_fnn = load_model_compat(fnn_path)
    X_fnn = normalize_fnn_batch(X_test_63)
    y_pred_fnn = np.argmax(model_fnn.predict(X_fnn, verbose=0), axis=1)
    metrics_fnn = evaluate_model_metrics(y_test, y_pred_fnn, class_names)
    t_fnn, _ = measure_inference_time(model_fnn, X_fnn[:1])
    params_fnn = count_parameters(model_fnn)
    size_fnn = get_model_size_mb(fnn_path)
    print(f"  Accuracy: {metrics_fnn['accuracy']*100:.2f}%, Params: {params_fnn:,}, Time: {t_fnn:.1f}ms")

    # --- 模型3: VGG16 ---
    print("\n[4/6] 评估 VGG16 融合模型 (对比模型)...")
    vgg_path = runtime.model_path('vgg')
    if os.path.exists(vgg_path):
        model_vgg = load_model_compat(vgg_path)
        X_img, X_lm, y_vgg = load_static_dual_dataset("test")
        X_img = preprocess_input(X_img)
        
        
        X_lm = standardize_vgg(X_lm)
        y_pred_vgg = np.argmax(model_vgg.predict([X_img, X_lm], verbose=0), axis=1)
        metrics_vgg = evaluate_model_metrics(y_vgg, y_pred_vgg, class_names)
        t_vgg, _ = measure_inference_time(model_vgg, [X_img[:1], X_lm[:1]])
        params_vgg = count_parameters(model_vgg)
        size_vgg = get_model_size_mb(vgg_path)
        print(f"  Accuracy: {metrics_vgg['accuracy']*100:.2f}%, Params: {params_vgg:,}, Time: {t_vgg:.1f}ms")
    else:
        print(f"  ⚠️ 未找到 VGG16 模型文件: {vgg_path}")
        print(f"  ⚠️ 请先运行 train_0_9_vgg16.py 训练VGG16模型，或将模型文件复制到项目根目录。")
        print(f"  ⚠️ 本次将跳过VGG16评估，使用占位数据。")
        metrics_vgg = {
            'accuracy': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0,
            'macro_f1': 0.0, 'weighted_f1': 0.0,
            'per_class': {str(i): {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0} for i in range(10)}
        }
        t_vgg, params_vgg, size_vgg = 0.0, 0, 0.0

    # --- 生成对比图表 ---
    print("\n[5/6] 生成静态手势对比图表...")
    model_names = ['1D-CNN (本文)', 'FNN', 'VGG16融合']
    all_metrics = [metrics_cnn, metrics_fnn, metrics_vgg]
    accuracies = [m['accuracy'] for m in all_metrics]
    param_list = [params_cnn, params_fnn, params_vgg]
    time_list = [t_cnn, t_fnn, t_vgg]
    size_list = [size_cnn, size_fnn, size_vgg]

    plot_accuracy_comparison_bar(model_names, accuracies,
        '静态手势识别（0-10）模型准确率对比', f'{OUTPUT_DIR}/static_accuracy_comparison.png')
    plot_per_class_f1_grouped(all_metrics, model_names, class_names,
        '静态手势识别（0-10）各类别F1值对比', f'{OUTPUT_DIR}/static_f1_per_class.png')
    plot_metrics_comparison_table(all_metrics, model_names, param_list, time_list, size_list,
        '静态手势识别模型综合指标对比', f'{OUTPUT_DIR}/static_metrics_table.png')
    plot_param_infer_comparison(model_names, param_list, time_list,
        '静态手势模型参数量与推理时间对比', f'{OUTPUT_DIR}/static_param_time.png')

    # --- 保存数据到JSON ---
    print("\n[6/6] 保存指标数据...")
    static_results = {
        'models': model_names,
        'metrics': {name: m for name, m in zip(model_names, all_metrics)},
        'params': {name: p for name, p in zip(model_names, param_list)},
        'infer_time_ms': {name: t for name, t in zip(model_names, time_list)},
        'model_size_mb': {name: s for name, s in zip(model_names, size_list)},
    }
    # 将 numpy 类型转为 Python 原生类型
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(f'{OUTPUT_DIR}/static_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(static_results), f, ensure_ascii=False, indent=2)
    print(f"  -> 已保存: {OUTPUT_DIR}/static_results.json")

    return static_results


def run_dynamic_comparison():
    """动态手势（10-20）三模型对比"""
    print("\n" + "="*60)
    print("  动态手势识别（10-20）对比实验")
    print("="*60)

    class_names = ACTIONS_DYNAMIC

    # --- 加载测试集 ---
    print("\n[1/6] 加载动态手势测试集...")
    x_test, y_test = load_dynamic_testset()

    # --- 模型1: ST-CNN (本文) ---
    print("\n[2/6] 评估 ST-CNN (本文模型)...")
    stcnn_path = runtime.model_path('stcnn')
    model_stcnn = build_st_cnn_model()
    model_stcnn.load_weights(stcnn_path)
    y_pred_stcnn = np.argmax(model_stcnn.predict(x_test, verbose=0), axis=1)
    metrics_stcnn = evaluate_model_metrics(y_test, y_pred_stcnn, class_names)
    t_stcnn, _ = measure_inference_time(model_stcnn, x_test[:1])
    params_stcnn = count_parameters(model_stcnn)
    size_stcnn = get_model_size_mb(stcnn_path)
    print(f"  Accuracy: {metrics_stcnn['accuracy']*100:.2f}%, Params: {params_stcnn:,}, Time: {t_stcnn:.1f}ms")

    # --- 模型2: GLT ---
    print("\n[3/6] 评估 GLT (对比模型)...")
    glt_path = runtime.model_path('glt')
    model_glt = build_glt_model()
    model_glt.load_weights(glt_path)
    y_pred_glt = np.argmax(model_glt.predict(x_test, verbose=0), axis=1)
    metrics_glt = evaluate_model_metrics(y_test, y_pred_glt, class_names)
    t_glt, _ = measure_inference_time(model_glt, x_test[:1])
    params_glt = count_parameters(model_glt)
    size_glt = get_model_size_mb(glt_path)
    print(f"  Accuracy: {metrics_glt['accuracy']*100:.2f}%, Params: {params_glt:,}, Time: {t_glt:.1f}ms")

    # --- 模型3: Transformer Lite ---
    print("\n[4/6] 评估 Transformer Lite (对比模型)...")
    trans_path = runtime.model_path('lite')
    model_trans = build_transformer_lite_model()
    model_trans.load_weights(trans_path)
    y_pred_trans = np.argmax(model_trans.predict(x_test, verbose=0), axis=1)
    metrics_trans = evaluate_model_metrics(y_test, y_pred_trans, class_names)
    t_trans, _ = measure_inference_time(model_trans, x_test[:1])
    params_trans = count_parameters(model_trans)
    size_trans = get_model_size_mb(trans_path)
    print(f"  Accuracy: {metrics_trans['accuracy']*100:.2f}%, Params: {params_trans:,}, Time: {t_trans:.1f}ms")

    # --- 生成对比图表 ---
    print("\n[5/6] 生成动态手势对比图表...")
    model_names = ['ST-CNN (本文)', 'GLT', 'Transformer Lite']
    all_metrics = [metrics_stcnn, metrics_glt, metrics_trans]
    accuracies = [m['accuracy'] for m in all_metrics]
    param_list = [params_stcnn, params_glt, params_trans]
    time_list = [t_stcnn, t_glt, t_trans]
    size_list = [size_stcnn, size_glt, size_trans]

    plot_accuracy_comparison_bar(model_names, accuracies,
        '动态手势识别（10-20）模型准确率对比', f'{OUTPUT_DIR}/dynamic_accuracy_comparison.png')
    plot_per_class_f1_grouped(all_metrics, model_names, class_names,
        '动态手势识别（10-20）各类别F1值对比', f'{OUTPUT_DIR}/dynamic_f1_per_class.png')
    plot_metrics_comparison_table(all_metrics, model_names, param_list, time_list, size_list,
        '动态手势识别模型综合指标对比', f'{OUTPUT_DIR}/dynamic_metrics_table.png')
    plot_param_infer_comparison(model_names, param_list, time_list,
        '动态手势模型参数量与推理时间对比', f'{OUTPUT_DIR}/dynamic_param_time.png')

    # --- 保存数据到JSON ---
    print("\n[6/6] 保存指标数据...")
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    dynamic_results = {
        'models': model_names,
        'metrics': {name: m for name, m in zip(model_names, all_metrics)},
        'params': {name: p for name, p in zip(model_names, param_list)},
        'infer_time_ms': {name: t for name, t in zip(model_names, time_list)},
        'model_size_mb': {name: s for name, s in zip(model_names, size_list)},
    }
    with open(f'{OUTPUT_DIR}/dynamic_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(dynamic_results), f, ensure_ascii=False, indent=2)
    print(f"  -> 已保存: {OUTPUT_DIR}/dynamic_results.json")

    return dynamic_results


# ===========================
# 入口
# ===========================
if __name__ == '__main__':
    print("=" * 60)
    print("  毕业设计 - 模型对比实验自动评估")
    print("=" * 60)

    static_res = run_static_comparison()
    dynamic_res = run_dynamic_comparison()

    print("\n" + "=" * 60)
    print("  全部对比实验完成！")
    print(f"  所有图表和数据已保存至: {OUTPUT_DIR}/")
    print("=" * 60)

    # 打印 Markdown 格式表格，可直接复制到论文
    print("\n\n===== 静态手势对比表格 (Markdown) =====\n")
    print("| 模型 | Accuracy | Precision | Recall | F1(Macro) | 参数量 | 推理时间(ms) |")
    print("|------|----------|-----------|--------|-----------|--------|-------------|")
    for name in static_res['models']:
        m = static_res['metrics'][name]
        p = static_res['params'][name]
        t = static_res['infer_time_ms'][name]
        print(f"| {name} | {m['accuracy']*100:.2f}% | {m['macro_precision']*100:.2f}% | {m['macro_recall']*100:.2f}% | {m['macro_f1']*100:.2f}% | {p:,} | {t:.1f} |")

    print("\n\n===== 动态手势对比表格 (Markdown) =====\n")
    print("| 模型 | Accuracy | Precision | Recall | F1(Macro) | 参数量 | 推理时间(ms) |")
    print("|------|----------|-----------|--------|-----------|--------|-------------|")
    for name in dynamic_res['models']:
        m = dynamic_res['metrics'][name]
        p = dynamic_res['params'][name]
        t = dynamic_res['infer_time_ms'][name]
        print(f"| {name} | {m['accuracy']*100:.2f}% | {m['macro_precision']*100:.2f}% | {m['macro_recall']*100:.2f}% | {m['macro_f1']*100:.2f}% | {p:,} | {t:.1f} |")
