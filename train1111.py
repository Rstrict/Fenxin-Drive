import os
from pathlib import Path

# =============== 1) 解决 Windows 常见 OpenMP 冲突 ===============
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =============== 2) 你只需要改这里 ===============
DATA_YAML = r"D:\BaiduNetdiskDownload\youtube_face_yolo_pose\images\train\data.yaml"  # 你的data.yaml路径
MODEL = "yolov8n-pose.pt"  # 也可以换 yolov8s-pose.pt
EPOCHS = 100
IMGSZ = 640
BATCH = 32
DEVICE = 0  # 0=第一张GPU；没有GPU可写 "cpu"
PROJECT = "pose"  # 输出目录
NAME = "face_pose27"   # 训练实验名
# ================================================

def main():
    # 防呆检查
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml 不存在：{data_path}")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "未安装 ultralytics，请先执行：pip install -U ultralytics"
        ) from e

    # =============== 3) 开始训练 ===============
    model = YOLO(MODEL)

    # 关键：amp=False 避免 AMP 检查加载损坏权重导致崩溃
    # val=False 先别让 COCO OKS 评估搞你（你的21脸部点很容易评估为0）
    results = model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        amp=False,
        val=False,
        project=PROJECT,
        name=NAME,
        plots=True,     # 会生成 labels.jpg / results.png
        workers=0       # Windows 下更稳，避免多进程读图问题
    )

    print("\n✅ 训练完成！")
    print("结果目录：", Path(PROJECT) / NAME)
    return results


if __name__ == "__main__":
    main()