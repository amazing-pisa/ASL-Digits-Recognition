import runtime
from compat import load_model_compat, read_image, normalize_fnn_batch, standardize_vgg, save_scaler, initialize_vgg_backbone
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

# =========================
# 0. 环境与配置
# =========================
DATASET_PATH = str(runtime.STATIC_DATASET)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


# =========================
# 1. 双模态数据加载
# =========================
def load_dual_dataset(split="train"):
    X_img, X_lm, y = [], [], []
    split_path = os.path.join(DATASET_PATH, split)
    print(f"[Info] Loading data from: {split_path}")
    if not os.path.exists(split_path): return np.array([]), np.array([]), np.array([])

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

            # MediaPipe 提取关键点
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 获取 21x3 坐标并展平为 63 维
                    lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                    # 图像预处理: 论文要求 resize 到 224x224
                    img_resized = cv2.resize(image_rgb, (224, 224))

                    X_img.append(img_resized)
                    X_lm.append(lm_array)
                    y.append(label)
                    break  # 只取第一只手

    return np.array(X_img, dtype=np.float32), np.array(X_lm, dtype=np.float32), np.array(y)


X_img_train, X_lm_train, y_train = load_dual_dataset("train")
X_img_val, X_lm_val, y_val = load_dual_dataset("valid")

# =========================
# 2. 预处理与标准化
# =========================
# 图像预处理 (ImageNet 均值方差标准化)
X_img_train = preprocess_input(X_img_train)
X_img_val = preprocess_input(X_img_val)

# 关键点标准化 (论文公式 4: 减均值除以标准差)
scaler = StandardScaler()
X_lm_train = scaler.fit_transform(X_lm_train)
save_scaler(scaler)
X_lm_val = scaler.transform(X_lm_val)

y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

# ==========================================
# 3. 构建 Full-VGG16 融合架构
# ==========================================
# 分支 1: VGG16 图像特征提取
img_input = Input(shape=(224, 224, 3), name="image_input")
vgg_base = VGG16(weights=None, include_top=False, input_tensor=img_input)
initialize_vgg_backbone(vgg_base)
vgg_base.trainable = False  # 冻结预训练权重

x_img = Flatten()(vgg_base.output)
# 提取 4096 维特征
x_img = Dense(4096, activation='relu')(x_img)
x_img = Dropout(0.5)(x_img)

# 分支 2: MediaPipe 关键点特征提取
lm_input = Input(shape=(63,), name="landmark_input")
# 映射到 256 维特征空间
x_lm = Dense(256, activation='relu')(lm_input)

# 融合层: 拼接 4096 维与 256 维 -> 4352 维
f_combined = Concatenate()([x_img, x_lm])

# 分类器输出层 (此处映射到 10 类，论文中为 29 类)
f_combined = Dropout(0.5)(f_combined)
outputs = Dense(10, activation='softmax', name="classifier_output")(f_combined)

model = Model(inputs=[img_input, lm_input], outputs=outputs, name="Full_VGG16_Fusion")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ==========================================
# 4. 训练与绘图
# ==========================================
callbacks = [
    ModelCheckpoint("full_vgg16_fusion_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

print("\n[Info] Training Full-VGG16 Fusion model...")
# 注意：这里输入是一个列表 [图像数据, 关键点数据]
history = model.fit([X_img_train, X_lm_train], y_train,
                    validation_data=([X_img_val, X_lm_val], y_val),
                    epochs=20, batch_size=32, callbacks=callbacks)


def plot_training_history(history_dict, save_path='full_vgg16_training_curves.png'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    epochs = range(1, len(history_dict['accuracy']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Accuracy
    ax1.plot(epochs, history_dict['accuracy'], 'b-', linewidth=2, label='训练集准确率')
    ax1.plot(epochs, history_dict['val_accuracy'], 'r--', linewidth=2, label='验证集准确率')
    ax1.set_title('Full-VGG16 准确率变化趋势', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.legend(fontsize=13)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Loss
    ax2.plot(epochs, history_dict['loss'], 'b-', linewidth=2, label='训练集损失')
    ax2.plot(epochs, history_dict['val_loss'], 'r--', linewidth=2, label='验证集损失')
    ax2.set_title('Full-VGG16 损失变化趋势', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.legend(fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ 训练曲线已保存至: {save_path}")


plot_training_history(history.history)
