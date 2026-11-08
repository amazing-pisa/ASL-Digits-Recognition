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
MODEL_PATH = runtime.model_path('cnn')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


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

# ===========================
# 4. 智能模型适配
# ===========================
print(f"\n[Info] 正在加载模型: {MODEL_PATH}")
model = load_model_compat(MODEL_PATH)
expected_shape = model.input_shape[1:]

if len(expected_shape) == 1 and expected_shape[0] == 63:
    X_test = X_test.reshape(-1, 63)
elif len(expected_shape) == 2 and expected_shape == (63, 1):
    X_test = X_test.reshape(-1, 63, 1)
elif len(expected_shape) == 2 and expected_shape == (21, 3):
    X_test = X_test.reshape(-1, 21, 3)

# ===========================
# 5. 预测与评估报告
# ===========================
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, digits=4))


# ===========================
# 6. 绘制高清混淆矩阵
# ===========================
def plot_academic_confusion_matrix(y_true, y_pred, save_path='hand_sign_confusion_matrix.png'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)],
                annot_kws={"size": 10})

    plt.title('归一化混淆矩阵 (0-9 静态手势)', fontsize=15, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('预测比例', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 混淆矩阵图已保存至: {save_path}")


plot_academic_confusion_matrix(y_test, y_pred)
