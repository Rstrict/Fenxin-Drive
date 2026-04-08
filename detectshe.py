import cv2
import numpy as np
import torch
from ultralytics_src.nn.tasks import PoseModel

torch.serialization.add_safe_globals([PoseModel])
from ultralytics_src import YOLO

from pathlib import Path
from ultralytics import YOLO

def camera_realtime_detection_gray():
    weight_path = Path(
        r'D:\Study\DeveloppingAI\chengxu\yolov8\runs\detect\runs\train\exp5\weights\best.pt'
    )

    print('模型路径 =', weight_path)
    print('文件存在吗 =', weight_path.exists())

    if not weight_path.exists():
        raise FileNotFoundError(f'模型不存在: {weight_path}')

    model = YOLO(str(weight_path))
    # 2. 打开摄像头（0为默认摄像头）
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    # 3. 创建显示窗口
    cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)

    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("警告：无法读取摄像头画面，退出...")
                break

            # 4. 核心步骤：彩色转黑白（灰度图）
            # cvtColor：COLOR_BGR2GRAY 将BGR彩色图转为单通道灰度图
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # YOLO模型要求输入为3通道图像，因此将单通道灰度图转回3通道（三个通道值相同）
            gray_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # 5. 用黑白画面执行检测（传入gray_3ch而非原始frame）
            results = model(gray_3ch, conf=0.25)

            # 6. 渲染检测结果到黑白画面上
            annotated_frame = results[0].plot()

            # 7. 显示黑白检测画面
            cv2.imshow("YOLOv8", annotated_frame)

            # 退出控制：按q或ESC键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("用户主动退出检测...")
                break
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放，程序结束")

if __name__ == "__main__":
    camera_realtime_detection_gray()