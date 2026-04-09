import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===================== 路径 =====================
NPZ_PATH = r"D:\下载\archive (1)\face_images.npz"
CSV_PATH = r"D:\下载\archive (1)\facial_keypoints.csv"
MODEL_PATH = r"D:\Study\DeveloppingAI\chengxu\yolov8\best_face_keypoint_model.pth"
IMG_SIZE = 96
# ==============================================


# ---------------------- 模型定义：必须和训练时完全一致 ----------------------
class FaceKeypointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 30),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ---------------------- 读取数据 ----------------------
data = np.load(NPZ_PATH)
images = data["face_images"]   # (96,96,N)
X = np.transpose(images, (2, 0, 1)).astype(np.float32) / 255.0
data.close()

df = pd.read_csv(CSV_PATH)
valid_mask = ~df.isna().any(axis=1)

X = X[valid_mask.values]
y = df[valid_mask].values.astype(np.float32)   # 先保留像素坐标，方便画真值

X = np.expand_dims(X, axis=1)  # [N,1,96,96]

# 和训练保持一致的划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("测试集大小:", len(X_test))

# ---------------------- 加载模型 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)

model = FaceKeypointModel().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("模型加载成功")


# ---------------------- 可视化函数 ----------------------
def show_keypoints(image_tensor, true_points=None, pred_points=None, title=""):
    image = image_tensor.squeeze().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

    if true_points is not None:
        true_points = np.array(true_points).reshape(-1, 2)
        plt.scatter(true_points[:, 0], true_points[:, 1], c="lime", s=20, marker="x", label="True")

    if pred_points is not None:
        pred_points = np.array(pred_points).reshape(-1, 2)
        plt.scatter(pred_points[:, 0], pred_points[:, 1], c="red", s=20, marker="o", label="Pred")

    if true_points is not None or pred_points is not None:
        plt.legend()

    plt.show()


# ---------------------- 随机预测并显示 ----------------------
num_show = 5
indices = random.sample(range(len(X_test)), num_show)

with torch.no_grad():
    for i, idx in enumerate(indices):
        img = X_test[idx].unsqueeze(0).to(device)  # [1,1,96,96]
        true_pts = y_test[idx].cpu().numpy()       # 真值本来就是像素坐标

        pred = model(img).squeeze(0).cpu().numpy()  # 这里应该是 0~1
        print(f"\nSample {i+1}")
        print("原始预测范围:", pred.min(), pred.max())

        pred_pts = pred * IMG_SIZE                  # 只乘一次
        pred_pts = np.clip(pred_pts, 0, IMG_SIZE - 1)

        show_keypoints(
            image_tensor=X_test[idx],
            true_points=true_pts,
            pred_points=pred_pts,
            title=f"Sample {i+1}"
        )