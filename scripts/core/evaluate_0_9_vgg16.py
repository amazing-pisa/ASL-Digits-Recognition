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
from tensorflow.keras.applications.vgg16 import preprocess_input

# ===========================
# 1. 基础配置
# ===========================
DATASET_PATH = str(runtime.STATIC_DATASET)
MODEL_PATH = runtime.model_path('vgg')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


# 注意：为了准确评估，理想情况下应当保存训练时的 scaler 实例(如使用 joblib)。
# 修复版仅从训练数据恢复标准化参数，不在测试集上拟合。
# 在实际工程中，请将 train.py 中的 scaler 导出并在此处加载。

def load_dual_dataset(split="test"):
    X_img, X_lm, y = [], [], []
    split_path = os.path.join(DATASET_PATH, split)
    print(f"[Info] Loading data from: {split_path}")

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
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    img_resized = cv2.resize(image_rgb, (224, 224))

                    X_img.append(img_resized)
                    X_lm.append(lm_array)
                    y.append(label)
                    break

    return np.array(X_img, dtype=np.float32), np.array(X_lm, dtype=np.float32), np.array(y, dtype=np.int32)


X_test_img, X_test_lm, y_test = load_dual_dataset("test")

# ===========================
# 2. 数据预处理
# ===========================
X_test_img = preprocess_input(X_test_img)

# 对测试集关键点进行近似标准化


X_test_lm = standardize_vgg(X_test_lm)

# ===========================
# 3. 预测与评估报告
# ===========================
print(f"\n[Info] 正在加载模型: {MODEL_PATH}")
model = load_model_compat(MODEL_PATH)

# 输入双模态数据
y_pred_prob = model.predict([X_test_img, X_test_lm], verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n===== Full-VGG16 Classification Report =====")
print(classification_report(y_test, y_pred, digits=4))


# ===========================
# 4. 绘制高清混淆矩阵
# ===========================
def plot_academic_confusion_matrix(y_true, y_pred, save_path='full_vgg16_confusion_matrix.png'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Oranges",  # 使用不同颜色与之前的模型区分
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)],
                annot_kws={"size": 10})

    plt.title('归一化混淆矩阵 (Full-VGG16 融合架构)', fontsize=15, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('预测比例', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 混淆矩阵图已保存至: {save_path}")


plot_academic_confusion_matrix(y_test, y_pred)
