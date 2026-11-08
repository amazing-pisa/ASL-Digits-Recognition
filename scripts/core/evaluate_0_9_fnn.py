import runtime
from compat import load_model_compat, read_image, normalize_fnn_batch, standardize_vgg, save_scaler, initialize_vgg_backbone
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ===========================
# 1. 基础配置
# ===========================
DATASET_PATH = str(runtime.STATIC_DATASET)

# 🌟 切换你要测试的模型路径
MODEL_PATH = runtime.model_path('fnn')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


# ===========================
# 2. 提取并严格归一化坐标 (默认提取 63 维)
# ===========================
def extract_landmarks(image_path):
    image = read_image(image_path)
    if image is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
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


# ===========================
# 3. 加载测试集
# ===========================
def load_dataset(split="test"):
    X, y, paths = [], [], []
    split_path = os.path.join(DATASET_PATH, split)
    print(f"[Info] Loading data from: {split_path}")
    if not os.path.exists(split_path): raise FileNotFoundError(f"Path not found: {split_path}")
    for label_folder in sorted(os.listdir(split_path)):
        label_path = os.path.join(split_path, label_folder)
        if not os.path.isdir(label_path): continue
        try:
            label = int(label_folder.replace("A", ""))
        except ValueError:
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
                paths.append(img_path)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


X_test, y_test, test_paths = load_dataset("test")
if len(X_test) == 0: raise ValueError("[Error] X_test is empty.")

# ===========================
# 4. 智能模型适配与维度重塑
# ===========================
print(f"\n[Info] 正在加载模型: {MODEL_PATH}")
model = load_model_compat(MODEL_PATH)

expected_shape = model.input_shape[1:]

if len(expected_shape) == 1 and expected_shape[0] == 42:
    print("[Info] 检测到文献 FNN 模型 (42维输入)，自动截取前 42 维 (剔除Z轴深度信息)...")
    X_test_2d = X_test.reshape(-1, 21, 3)[:, :, :2]
    X_test = normalize_fnn_batch(X_test)
elif len(expected_shape) == 1 and expected_shape[0] == 63:
    print("[Info] 检测到标准 MLP 模型，数据维度自动塑形为: (Samples, 63)")
    X_test = X_test.reshape(-1, 63)
elif len(expected_shape) == 2 and expected_shape == (63, 1):
    print("[Info] 检测到原始 1D-CNN 模型，数据维度自动塑形为: (Samples, 63, 1)")
    X_test = X_test.reshape(-1, 63, 1)
elif len(expected_shape) == 2 and expected_shape == (21, 3):
    print("[Info] 检测到优化版 1D-CNN 或空间模型，数据维度自动塑形为: (Samples, 21, 3)")
    X_test = X_test.reshape(-1, 21, 3)
else:
    print(f"[Warning] 未知模型输入维度: {expected_shape}，尝试保持默认形状。")

# ===========================
# 5. 预测与打印文本报告
# ===========================
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, digits=4))


# ===========================
# 6. 绘制学术级混淆矩阵
# ===========================
def plot_academic_confusion_matrix(y_true, y_pred, classes, save_path='fnn_confusion_matrix.png'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8), dpi=300)
    # 使用橙色系以区分其他模型
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 11}, cbar_kws={'label': '预测比例'})

    plt.title('Paper FNN 测试集混淆矩阵 (归一化)', fontsize=16, pad=20)
    plt.ylabel('真实标签 (True Label)', fontsize=14)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=14)
    cbar = plt.gcf().axes[-1]
    cbar.yaxis.label.set_size(14)
    cbar.tick_params(labelsize=12)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 混淆矩阵已保存至: {save_path}")


# 调用绘图函数，0-9 的标签
classes_0_9 = [str(i) for i in range(10)]
plot_academic_confusion_matrix(y_test, y_pred, classes_0_9)
