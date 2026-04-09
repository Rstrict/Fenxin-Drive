# -------------------------读取
import os
import glob
import cv2
import yaml
import shutil
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# 1. 路径配置
# =========================
NPZ_ROOT = r"D:\BaiduNetdiskDownload\archive"
OUT_ROOT = r"D:\BaiduNetdiskDownload\youtube_face_yolo_pose"

CLASS_ID = 0
CLASS_NAME = "face"

VAL_RATIO = 0.2
RANDOM_SEED = 42
DEFAULT_VIS = 2          # YOLO pose 可见性: 0/1/2
MIN_VALID_KPTS = 3       # 至少有效关键点数
BBOX_MARGIN_RATIO = 0.05
IMG_EXT = ".jpg"


# =========================
# 2. 基础工具
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_output_dirs(root):
    if os.path.exists(root):
        shutil.rmtree(root)

    ensure_dir(os.path.join(root, "images", "train"))
    ensure_dir(os.path.join(root, "images", "val"))
    ensure_dir(os.path.join(root, "labels", "train"))
    ensure_dir(os.path.join(root, "labels", "val"))


def find_npz_files(npz_root):
    files = glob.glob(os.path.join(npz_root, "**", "*.npz"), recursive=True)
    return sorted(files)


# =========================
# 3. 数据格式归一化
# =========================
def normalize_landmarks_array(y):
    """
    统一输出为 (N, K, 2)

    支持:
    - (K, 2)
    - (N, K, 2)
    - (N, 2, K)
    - (K, 2, N)
    - (2, K, N)
    - (N, K*2)
    """
    y = np.asarray(y)

    # (K,2)
    if y.ndim == 2:
        if y.shape[-1] == 2:
            return y[None, ...].astype(np.float32)

        if y.shape[0] == 2:
            # (2,K) -> (1,K,2)
            return y.T[None, ...].astype(np.float32)

        if y.shape[1] % 2 == 0:
            # (N,K*2) -> (N,K,2)
            k = y.shape[1] // 2
            return y.reshape(y.shape[0], k, 2).astype(np.float32)

    if y.ndim == 3:
        # (N,K,2)
        if y.shape[-1] == 2:
            return y.astype(np.float32)

        # 可能是 (N,2,K) 或 (K,2,N)
        if y.shape[1] == 2:
            if y.shape[2] > y.shape[0]:
                # (K,2,N) -> (N,K,2)
                return np.transpose(y, (2, 0, 1)).astype(np.float32)
            else:
                # (N,2,K) -> (N,K,2)
                return np.transpose(y, (0, 2, 1)).astype(np.float32)

        # (2,K,N) -> (N,K,2)
        if y.shape[0] == 2:
            return np.transpose(y, (2, 1, 0)).astype(np.float32)

    raise ValueError(f"不支持的 landmarks2D 数组形状: {y.shape}")


def normalize_image_array(X, expected_n=None):
    """
    统一输出为:
    - 灰度: (N,H,W)
    - 彩图: (N,H,W,C)

    支持常见输入:
    - (H,W)
    - (H,W,C)
    - (C,H,W)
    - (N,H,W)
    - (N,H,W,C)
    - (N,C,H,W)
    - (H,W,C,N)
    - (C,H,W,N)
    - (H,W,N,C)
    """
    X = np.asarray(X)

    # 单张灰度图
    if X.ndim == 2:
        return X[None, ...]

    # 3维
    if X.ndim == 3:
        # 单张 HWC
        if X.shape[-1] in [1, 3, 4]:
            return X[None, ...]

        # 单张 CHW
        if X.shape[0] in [1, 3, 4] and X.shape[1] > 4 and X.shape[2] > 4:
            return np.transpose(X, (1, 2, 0))[None, ...]

        # 默认视为 (N,H,W)
        return X

    # 4维
    if X.ndim == 4:
        if expected_n is not None:
            # (N,H,W,C)
            if X.shape[0] == expected_n and X.shape[-1] in [1, 3, 4]:
                return X

            # (N,C,H,W)
            if X.shape[0] == expected_n and X.shape[1] in [1, 3, 4]:
                return np.transpose(X, (0, 2, 3, 1))

            # (H,W,C,N)
            if X.shape[-1] == expected_n and X.shape[2] in [1, 3, 4]:
                return np.transpose(X, (3, 0, 1, 2))

            # (C,H,W,N)
            if X.shape[-1] == expected_n and X.shape[0] in [1, 3, 4]:
                return np.transpose(X, (3, 1, 2, 0))

            # (H,W,N,C)
            if X.shape[2] == expected_n and X.shape[-1] in [1, 3, 4]:
                return np.transpose(X, (2, 0, 1, 3))

        # 没有 expected_n 时的兜底
        if X.shape[-1] in [1, 3, 4]:
            return X

        if X.shape[1] in [1, 3, 4]:
            return np.transpose(X, (0, 2, 3, 1))

        if X.shape[2] in [1, 3, 4]:
            return np.transpose(X, (3, 0, 1, 2))

    raise ValueError(f"不支持的 colorImages 数组形状: {X.shape}")


def to_uint8_image(img):
    img = np.asarray(img)

    # (H,W,1) -> (H,W)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]

    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# =========================
# 4. 关键点 -> bbox
# =========================
def bbox_from_keypoints(kpts_xy, img_w, img_h, margin_ratio=0.05):
    xs = kpts_xy[:, 0]
    ys = kpts_xy[:, 1]

    valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (ys >= 0)
    if valid.sum() < MIN_VALID_KPTS:
        return None

    x_min = xs[valid].min()
    x_max = xs[valid].max()
    y_min = ys[valid].min()
    y_max = ys[valid].max()

    bw = max(2.0, x_max - x_min)
    bh = max(2.0, y_max - y_min)

    mx = bw * margin_ratio
    my = bh * margin_ratio

    x_min = max(0.0, x_min - mx)
    y_min = max(0.0, y_min - my)
    x_max = min(img_w - 1.0, x_max + mx)
    y_max = min(img_h - 1.0, y_max + my)

    if x_max <= x_min or y_max <= y_min:
        return None

    return [x_min, y_min, x_max, y_max]


# =========================
# 5. 生成 YOLO pose 标签
# =========================
def build_yolo_pose_label(kpts_xy, img_w, img_h):
    bbox_xyxy = bbox_from_keypoints(kpts_xy, img_w, img_h, BBOX_MARGIN_RATIO)
    if bbox_xyxy is None:
        return None

    x_min, y_min, x_max, y_max = bbox_xyxy
    bw = x_max - x_min
    bh = y_max - y_min

    if bw <= 1 or bh <= 1:
        return None

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw /= img_w
    bh /= img_h

    parts = [
        str(CLASS_ID),
        f"{cx:.6f}", f"{cy:.6f}",
        f"{bw:.6f}", f"{bh:.6f}"
    ]

    num_kpts = kpts_xy.shape[0]
    for i in range(num_kpts):
        x, y = kpts_xy[i]

        if np.isfinite(x) and np.isfinite(y) and x >= 0 and y >= 0:
            x_n = np.clip(x / img_w, 0.0, 1.0)
            y_n = np.clip(y / img_h, 0.0, 1.0)
            v = DEFAULT_VIS
        else:
            x_n, y_n, v = 0.0, 0.0, 0

        parts.extend([f"{x_n:.6f}", f"{y_n:.6f}", str(v)])

    return " ".join(parts)


# =========================
# 6. 读取所有样本
# =========================
def load_all_samples():
    npz_files = find_npz_files(NPZ_ROOT)
    print(f"找到 npz 文件数量: {len(npz_files)}")

    all_samples = []
    skipped_files = []

    for idx, npz_file in enumerate(npz_files, 1):
        try:
            data = np.load(npz_file, allow_pickle=True)

            required_keys = ["colorImages", "landmarks2D"]
            if not all(k in data.files for k in required_keys):
                print(f"[跳过] 缺少必要键: {npz_file}, keys={data.files}")
                skipped_files.append(npz_file)
                continue

            raw_X = data["colorImages"]
            raw_Y = data["landmarks2D"]

            Y = normalize_landmarks_array(raw_Y)
            X = normalize_image_array(raw_X, expected_n=Y.shape[0])

            n = min(len(X), len(Y))
            X = X[:n]
            Y = Y[:n]

            stem = Path(npz_file).stem

            if idx == 1:
                print("\n===== 首个样本文件调试信息 =====")
                print("文件:", npz_file)
                print("raw colorImages shape:", np.asarray(raw_X).shape)
                print("raw landmarks2D shape:", np.asarray(raw_Y).shape)
                if "boundingBox" in data.files:
                    print("raw boundingBox shape:", np.asarray(data["boundingBox"]).shape)
                print("normalized colorImages shape:", X.shape)
                print("normalized landmarks2D shape:", Y.shape)
                print("关键点数量 K =", Y.shape[1])
                print("================================\n")

            for i in range(n):
                all_samples.append({
                    "base_name": f"{stem}_{i:05d}",
                    "image": X[i],
                    "kpts_xy": Y[i].astype(np.float32)
                })

            print(f"[{idx}/{len(npz_files)}] 已读取: {npz_file} | 样本数: {n}")

        except Exception as e:
            print(f"[错误] {npz_file} -> {e}")
            skipped_files.append(npz_file)

    print(f"\n总样本数: {len(all_samples)}")
    print(f"跳过文件数: {len(skipped_files)}")
    return all_samples


# =========================
# 7. 写出 YOLO 数据集
# =========================
def write_dataset(samples):
    reset_output_dirs(OUT_ROOT)

    if len(samples) < 2:
        train_samples = samples
        val_samples = []
    else:
        train_samples, val_samples = train_test_split(
            samples,
            test_size=VAL_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True
        )

    def save_subset(subset, split_name):
        img_dir = os.path.join(OUT_ROOT, "images", split_name)
        lbl_dir = os.path.join(OUT_ROOT, "labels", split_name)

        saved = 0
        skipped = 0

        for item in subset:
            name = item["base_name"]
            img = to_uint8_image(item["image"])
            kpts_xy = item["kpts_xy"]

            if img.ndim == 2:
                h, w = img.shape
            elif img.ndim == 3:
                h, w = img.shape[:2]
            else:
                skipped += 1
                continue

            label_line = build_yolo_pose_label(kpts_xy, w, h)
            if label_line is None:
                skipped += 1
                continue

            img_path = os.path.join(img_dir, name + IMG_EXT)
            txt_path = os.path.join(lbl_dir, name + ".txt")

            ok = cv2.imwrite(img_path, img)
            if not ok:
                print(f"[失败] 图片保存失败: {img_path}")
                skipped += 1
                continue

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(label_line + "\n")

            saved += 1

        return saved, skipped

    train_saved, train_skipped = save_subset(train_samples, "train")
    val_saved, val_skipped = save_subset(val_samples, "val")

    print(f"\n训练集: 保存 {train_saved}，跳过 {train_skipped}")
    print(f"验证集: 保存 {val_saved}，跳过 {val_skipped}")


# =========================
# 8. 生成 data.yaml
# =========================
def write_yaml(samples):
    if len(samples) == 0:
        return

    num_kpts = samples[0]["kpts_xy"].shape[0]
    yaml_path = os.path.join(OUT_ROOT, "data.yaml")

    data_yaml = {
        "path": OUT_ROOT.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [int(num_kpts), 3],
        "names": {
            0: CLASS_NAME
        }
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"已生成: {yaml_path}")
    print(f"kpt_shape: [{num_kpts}, 3]")


# =========================
# 9. 主函数
# =========================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    samples = load_all_samples()
    if len(samples) == 0:
        print("没有可用样本，转换终止。")
        return

    write_dataset(samples)
    write_yaml(samples)

    print("\n转换完成。输出目录：")
    print(OUT_ROOT)

    print("\n训练命令示例：")
    print(
        f'yolo pose train model=yolov8n-pose.pt '
        f'data="{os.path.join(OUT_ROOT, "data.yaml")}" '
        f'imgsz=96 epochs=100 batch=32'
    )


if __name__ == "__main__":
    main()




# # ----------------------训练
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# import warnings
# warnings.filterwarnings("ignore")
#
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader
#
# # ===================== 路径配置 =====================
# NPZ_PATH = r"D:\下载\archive (1)\face_images.npz"
# CSV_PATH = r"D:\下载\archive (1)\facial_keypoints.csv"
# BEST_MODEL_PATH = r"D:\Study\DeveloppingAI\chengxu\yolov8\best_face_keypoint_model.pth"
# # ==================================================
#
# # ===================== 超参数 ======================
# IMG_SIZE = 96
# BATCH_SIZE = 64
# EPOCHS = 50
# LR = 1e-3
# RANDOM_SEED = 42
# # ==================================================
#
# # ---------------------- 固定随机种子 ----------------------
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
# set_seed(RANDOM_SEED)
#
# # ---------------------- 选择设备 ----------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("当前设备:", device)
#
# # ---------------------- 1. 读取图像 ----------------------
# data = np.load(NPZ_PATH)
# images = data["face_images"]   # 原始形状一般是 (96, 96, N)
#
# # 转成 [N, 1, 96, 96]
# X = np.transpose(images, (2, 0, 1)).astype(np.float32)   # [N, 96, 96]
# X = X / 255.0
# data.close()
#
# # ---------------------- 2. 读取标签 ----------------------
# df = pd.read_csv(CSV_PATH)
#
# print("标签形状:", df.shape)
# print("总缺失值数量:", df.isna().sum().sum())
#
# # 删除含缺失值的样本
# valid_mask = ~df.isna().any(axis=1)
#
# print("原始样本数:", len(df))
# print("可用样本数:", valid_mask.sum())
# print("被删除样本数:", len(df) - valid_mask.sum())
#
# # 图像和标签同步筛选
# X = X[valid_mask.values]                              # [N, 96, 96]
# y = df[valid_mask].values.astype(np.float32)         # [N, 30]
#
# # ---------------------- 3. 坐标归一化 ----------------------
# # 原始关键点坐标范围大致在 0~96，这里归一化到 0~1
# y = y / IMG_SIZE
#
# # 给图像增加通道维度
# X = np.expand_dims(X, axis=1)                        # [N, 1, 96, 96]
#
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("X 是否存在 NaN:", np.isnan(X).any())
# print("y 是否存在 NaN:", np.isnan(y).any())
#
# # ---------------------- 4. 划分训练/验证集 ----------------------
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=RANDOM_SEED
# )
#
# print("训练集大小:", len(X_train))
# print("验证集大小:", len(X_val))
#
# # 转成 Tensor
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_val   = torch.tensor(X_val, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# y_val   = torch.tensor(y_val, dtype=torch.float32)
#
# # ---------------------- 5. DataLoader ----------------------
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)
#
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=torch.cuda.is_available()
# )
#
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=torch.cuda.is_available()
# )
#
# # ---------------------- 6. 模型 ----------------------
# class FaceKeypointModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv_layers = nn.Sequential(
#             # 输入: [B, 1, 96, 96]
#             nn.Conv2d(1, 32, kernel_size=3),   # -> [B, 32, 94, 94]
#             nn.ReLU(),
#             nn.MaxPool2d(2),                   # -> [B, 32, 47, 47]
#
#             nn.Conv2d(32, 64, kernel_size=3),  # -> [B, 64, 45, 45]
#             nn.ReLU(),
#             nn.MaxPool2d(2),                   # -> [B, 64, 22, 22]
#
#             nn.Conv2d(64, 128, kernel_size=3), # -> [B, 128, 20, 20]
#             nn.ReLU(),
#             nn.MaxPool2d(2),                   # -> [B, 128, 10, 10]
#
#             nn.Conv2d(128, 256, kernel_size=3),# -> [B, 256, 8, 8]
#             nn.ReLU(),
#             nn.MaxPool2d(2),                   # -> [B, 256, 4, 4]
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#
#             nn.Linear(512, 30),
#             nn.Sigmoid()   # 因为标签已经归一化到 0~1
#         )
#
#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x
#
# model = FaceKeypointModel().to(device)
#
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)
#
# # 可选：学习率调度器
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.5,
#     patience=5
# )
#
# # ---------------------- 7. 训练与验证 ----------------------
# best_val_loss = float("inf")
# best_epoch = -1
#
# print("\n开始训练...\n")
#
# for epoch in range(EPOCHS):
#     # ===== 训练 =====
#     model.train()
#     train_loss_sum = 0.0
#     train_samples = 0
#
#     for batch_imgs, batch_labels in train_loader:
#         batch_imgs = batch_imgs.to(device, non_blocking=True)
#         batch_labels = batch_labels.to(device, non_blocking=True)
#
#         optimizer.zero_grad()
#
#         preds = model(batch_imgs)
#         loss = criterion(preds, batch_labels)
#
#         loss.backward()
#         optimizer.step()
#
#         batch_size_now = batch_imgs.size(0)
#         train_loss_sum += loss.item() * batch_size_now
#         train_samples += batch_size_now
#
#     avg_train_loss = train_loss_sum / train_samples
#
#     # ===== 验证 =====
#     model.eval()
#     val_loss_sum = 0.0
#     val_samples = 0
#
#     with torch.no_grad():
#         for batch_imgs, batch_labels in val_loader:
#             batch_imgs = batch_imgs.to(device, non_blocking=True)
#             batch_labels = batch_labels.to(device, non_blocking=True)
#
#             preds = model(batch_imgs)
#             loss = criterion(preds, batch_labels)
#
#             batch_size_now = batch_imgs.size(0)
#             val_loss_sum += loss.item() * batch_size_now
#             val_samples += batch_size_now
#
#     avg_val_loss = val_loss_sum / val_samples
#
#     # 学习率调度
#     scheduler.step(avg_val_loss)
#
#     current_lr = optimizer.param_groups[0]["lr"]
#
#     print(
#         f"Epoch [{epoch+1}/{EPOCHS}] | "
#         f"Train Loss: {avg_train_loss:.6f} | "
#         f"Val Loss: {avg_val_loss:.6f} | "
#         f"LR: {current_lr:.6f}"
#     )
#
#     # ===== 保存最佳模型 =====
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         best_epoch = epoch + 1
#
#         torch.save({
#             "epoch": best_epoch,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "best_val_loss": best_val_loss,
#             "img_size": IMG_SIZE,
#             "num_keypoints": 15,
#             "output_dim": 30
#         }, BEST_MODEL_PATH)
#
#         print(f"  -> 已保存最佳模型到: {BEST_MODEL_PATH}")
#
# print("\n训练完成")
# print(f"最佳轮次: Epoch {best_epoch}")
# print(f"最佳验证损失: {best_val_loss:.6f}")