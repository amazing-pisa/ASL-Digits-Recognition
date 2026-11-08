import runtime
from compat import load_model_compat, read_image, normalize_fnn_batch, standardize_vgg, save_scaler, initialize_vgg_backbone
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# 替换为你的数据集路径
DATASET_PATH = str(runtime.STATIC_DATASET)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


# ==========================================
# 1. 数据提取 (严格复现：只提取 X 和 Y 坐标，42维)
# ==========================================
def extract_landmarks_2d(image_path):
    image = read_image(image_path)
    if image is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_array = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            wrist = lm_array[0]
            lm_array = lm_array - wrist
            max_dist = np.max(np.linalg.norm(lm_array, axis=1))
            if max_dist > 0:
                lm_array = lm_array / max_dist
            return lm_array.flatten()
    return None


def load_dataset(split="train"):
    X, y = [], []
    split_path = os.path.join(DATASET_PATH, split)
    print(f"[Info] Loading data from: {split_path}")
    if not os.path.exists(split_path): return np.array([]), np.array([])
    for label_folder in sorted(os.listdir(split_path)):
        label_path = os.path.join(split_path, label_folder)
        if not os.path.isdir(label_path): continue
        try:
            label = int(label_folder.replace("A", ""))
        except ValueError:
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            landmarks = extract_landmarks_2d(img_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y)


X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("valid")
X_test, y_test = load_dataset("test")

X_train = X_train.reshape(-1, 42)
X_val = X_val.reshape(-1, 42)
X_test = X_test.reshape(-1, 42)

y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# ==========================================
# 2. 构建文献描述的 FNN 模型
# ==========================================
model = Sequential([
    Input(shape=(42,), name="Input_Layer"),
    Dense(32, activation='relu', name="Hidden_Layer_1"),
    Dropout(0.4, name="Dropout_Layer"),
    Dense(10, activation='relu', name="Hidden_Layer_2"),
    Dense(10, activation='softmax', name="Output_Layer")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint("paper_fnn_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
]

print("\n[Info] 开始训练文献级 FNN 模型...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=callbacks)


# ==========================================
# 3. 绘制并保存训练曲线
# ==========================================
def plot_training_history(history_dict, save_path='fnn_training_curves.png'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    epochs = range(1, len(history_dict['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Accuracy 曲线
    ax1.plot(epochs, history_dict['accuracy'], 'b-', linewidth=2, label='训练集准确率 (Train Acc)')
    ax1.plot(epochs, history_dict['val_accuracy'], 'r--', linewidth=2, label='验证集准确率 (Val Acc)')
    ax1.set_title('Paper FNN 模型准确率变化趋势', fontsize=16)
    ax1.set_xlabel('迭代轮数 (Epochs)', fontsize=14)
    ax1.set_ylabel('准确率 (Accuracy)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=13)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Loss 曲线
    ax2.plot(epochs, history_dict['loss'], 'b-', linewidth=2, label='训练集损失 (Train Loss)')
    ax2.plot(epochs, history_dict['val_loss'], 'r--', linewidth=2, label='验证集损失 (Val Loss)')
    ax2.set_title('Paper FNN 模型损失函数变化趋势', fontsize=16)
    ax2.set_xlabel('迭代轮数 (Epochs)', fontsize=14)
    ax2.set_ylabel('损失 (Loss)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 训练曲线已保存至: {save_path}")


plot_training_history(history.history)
