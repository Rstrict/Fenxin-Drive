import cv2
import numpy as np
from ultralytics import YOLO
import math

# ================= 修改这里 =================
MODEL_PATH = r"D:\Study\DeveloppingAI\chengxu\yolov8\runs\pose\runs\pose\pose215\face_pose215\weights\best.pt"
SOURCE = r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3\images"
# ============================================

# 关键点索引
RIGHT_EYE = list(range(0, 6))
LEFT_EYE = list(range(6, 12))
MOUTH = list(range(12, 20))
NOSE = 20


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_EAR(eye_points):
    # 眼睛 6点：0-5
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_MAR(mouth_points):
    # 嘴巴 8点：12-19
    A = euclidean(mouth_points[1], mouth_points[7])
    B = euclidean(mouth_points[2], mouth_points[6])
    C = euclidean(mouth_points[3], mouth_points[5])
    D = euclidean(mouth_points[0], mouth_points[4])
    return (A + B + C) / (3.0 * D + 1e-6)


def draw_closed_loop(img, points, color):
    for i in range(len(points)):
        p1 = tuple(map(int, points[i]))
        p2 = tuple(map(int, points[(i + 1) % len(points)]))
        cv2.line(img, p1, p2, color, 2)


def main():
    model = YOLO(MODEL_PATH)
    results = model.predict(SOURCE, save=False, conf=0.3)

    for r in results:
        img = r.orig_img.copy()

        if r.keypoints is None:
            continue

        for kpts in r.keypoints.xy:
            pts = kpts.cpu().numpy()

            # 右眼
            right_eye_pts = pts[RIGHT_EYE]
            draw_closed_loop(img, right_eye_pts, (0, 255, 0))

            # 左眼
            left_eye_pts = pts[LEFT_EYE]
            draw_closed_loop(img, left_eye_pts, (255, 0, 0))

            # 嘴巴
            mouth_pts = pts[MOUTH]
            draw_closed_loop(img, mouth_pts, (0, 0, 255))

            # 鼻子
            nose_pt = tuple(map(int, pts[NOSE]))
            cv2.circle(img, nose_pt, 4, (255, 255, 0), -1)

            # 计算 EAR
            ear_right = compute_EAR(right_eye_pts)
            ear_left = compute_EAR(left_eye_pts)
            ear = (ear_right + ear_left) / 2.0

            # 计算 MAR
            mar = compute_MAR(mouth_pts)

            cv2.putText(img, f"EYE: {ear:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(img, f"MAR: {mar:.3f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Pose21", img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()