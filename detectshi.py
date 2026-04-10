import cv2
from ultralytics import YOLO

def detect_video(model_path, video_path, save_path="output_detect.mp4"):
    # 1. 加载训练好的模型
    model = YOLO(model_path)

    # 2. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    # 3. 获取视频基本信息（帧率、分辨率、编码格式）
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码格式（MP4）

    # 4. 创建视频写入器，用于保存检测后的视频
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # 5. 创建显示窗口（英文名称避免乱码）
    cv2.namedWindow("YOLOv8 Video Detection", cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            # 读取视频的一帧
            ret, frame = cap.read()
            if not ret:
                print("视频读取完毕或出错，退出...")
                break

            # ========== 核心：模型推理 ==========
            # conf=0.25 过滤低置信度检测框，可根据需求调整
            results = model(frame, conf=0.25)

            # 渲染检测结果（自动画框、类别、置信度）
            annotated_frame = results[0].plot()

            # ========== 保存+显示 ==========
            out.write(annotated_frame)  # 写入结果视频
            cv2.imshow("YOLOv8 Video Detection", annotated_frame)  # 实时显示

            # 按 q 或 ESC 键提前退出
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                print("用户主动退出检测...")
                break
    finally:
        # 释放所有资源（必须执行）
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"检测完成！结果视频已保存到：{save_path}")

# ========== 运行检测 ==========
if __name__ == "__main__":
    # 替换为你的模型路径、视频路径
    detect_video(
        model_path=r"D:\Study\DeveloppingAI\chengxu\yolov8\runs\pose\pose\face_pose27\weights\best.pt",
        video_path="D:/Study/DeveloppingAI/Detect/ceshi/video/12.mp4",
        save_path="detected_result2.mp4"  # 输出视频路径
    )