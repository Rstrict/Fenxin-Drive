import cv2
import numpy as np
from ultralytics import YOLO

# ===================== 你只改这里 =====================
MODEL_PATH = r"D:\Study\DeveloppingAI\chengxu\yolov8\runs\pose\runs\pose\face_pose26\weights\best.pt"
VIDEO_PATH = r"D:\Study\DeveloppingAI\Detect\ceshi\video\13.mp4"  # 改成你的视频
# =====================================================

# 关键点索引
RIGHT_EYE = list(range(0, 6))
LEFT_EYE = list(range(6, 12))
MOUTH = list(range(12, 20))
NOSE = 20

# 阈值
EAR_THRESH = 0.25
MAR_THRESH = 0.60
CONSEC_FRAMES = 5
FATIGUE_EYE_TIME = 1.5  # 连续闭眼>=1.5秒 => 疲劳报警

# 播放控制
STEP_FRAMES = 10
DEFAULT_WAIT = 1

# 放大查看
ZOOM_MIN = 1.0
ZOOM_MAX = 6.0
ZOOM_STEP = 0.25


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_EAR(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_MAR(mouth):
    A = euclidean(mouth[1], mouth[7])
    B = euclidean(mouth[2], mouth[6])
    C = euclidean(mouth[3], mouth[5])
    D = euclidean(mouth[0], mouth[4])
    return (A + B + C) / (3.0 * D + 1e-6)


def draw_loop(img, pts, color, thickness=2):
    pts = pts.astype(int)
    for i in range(len(pts)):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % len(pts)])
        cv2.line(img, p1, p2, color, thickness)


def choose_driver_index(boxes_xyxy):
    centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    return int(np.argmax(centers_x))


def overlay_help(frame):
    lines = [
        "Keys:",
        "  SPACE : Pause/Resume",
        "  A : Backward 10f   |  D : Forward 10f",
        "  J : -100f          |  L : +100f",
        "  Z : Zoom on/off    |  + / - : Zoom in/out",
        "  H : Help on/off",
        "  Q / ESC : Quit",
    ]
    y = 25
    for s in lines:
        cv2.putText(frame, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 22


def apply_zoom(frame, center_xy, zoom_scale):
    """
    以 center_xy 为中心，从原图裁剪一个更小窗口，再resize回原尺寸，实现放大效果
    zoom_scale=1.0 表示不放大；2.0表示放大2倍（裁剪尺寸变为原来的1/2）
    """
    if zoom_scale <= 1.0 or center_xy is None:
        return frame

    h, w = frame.shape[:2]
    cx, cy = center_xy

    crop_w = int(w / zoom_scale)
    crop_h = int(h / zoom_scale)

    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # 边界裁剪
    if x1 < 0:
        x1 = 0
        x2 = crop_w
    if y1 < 0:
        y1 = 0
        y2 = crop_h
    if x2 > w:
        x2 = w
        x1 = w - crop_w
    if y2 > h:
        y2 = h
        y1 = h - crop_h

    # 防止 crop_w/crop_h 太大导致负数
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame

    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 视频打不开：", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    blink_counter = 0
    blink_total = 0

    paused = False
    show_help = True

    zoom_on = False
    zoom_scale = 2.0  # 默认放大2倍

    cur = 0
    frame = None

    # 用于放大中心：优先使用“驾驶员bbox中心”
    last_zoom_center = None

    while True:
        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ret, frame = cap.read()
            if not ret:
                break
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            if frame is None:
                # 第一次就暂停时，强制读一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
                ret, frame = cap.read()
                if not ret:
                    break
                cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        vis = frame.copy()

        # 推理
        r = model.predict(vis, verbose=False, conf=0.25)[0]

        # 默认本帧没有可用中心
        this_center = None

        if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0 and r.keypoints.xy is not None:
            if r.keypoints.xy.shape[0] > 0:
                boxes = r.boxes.xyxy.cpu().numpy()          # (N,4)
                kpts_all = r.keypoints.xy.cpu().numpy()     # (N,21,2)

                idx = choose_driver_index(boxes)
                pts = kpts_all[idx]
                bx1, by1, bx2, by2 = boxes[idx]
                this_center = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                last_zoom_center = this_center  # 更新放大中心（用于没检测到时继续用上一次）

                right_eye = pts[RIGHT_EYE]
                left_eye = pts[LEFT_EYE]
                mouth = pts[MOUTH]
                nose = pts[NOSE]

                ear = (compute_EAR(right_eye) + compute_EAR(left_eye)) / 2.0
                mar = compute_MAR(mouth)

                # 眨眼计数（只在播放时更新）
                if not paused:
                    if ear < EAR_THRESH:
                        blink_counter += 1
                    else:
                        if blink_counter >= CONSEC_FRAMES:
                            blink_total += 1
                        blink_counter = 0

                # 连线（画在原vis上，后面再做放大也能看清）
                draw_loop(vis, right_eye, (0, 255, 0), 2)      # 右眼绿
                draw_loop(vis, left_eye, (255, 0, 0), 2)       # 左眼蓝
                draw_loop(vis, mouth, (0, 0, 255), 2)          # 嘴红
                cv2.circle(vis, tuple(nose.astype(int)), 4, (0, 255, 255), -1)  # 鼻子黄

                # 数值显示
                cv2.putText(vis, f"EAR: {ear:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis, f"MAR: {mar:.3f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(vis, f"Blinks: {blink_total}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if mar > MAR_THRESH:
                    cv2.putText(vis, "MOUTH OPEN!", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # 疲劳报警：连续闭眼
                if blink_counter >= int(FATIGUE_EYE_TIME * fps):
                    cv2.putText(vis, "FATIGUE DRIVING !!!", (20, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

                # 在原图上画出 bbox 中心（方便看放大中心）
                cv2.circle(vis, this_center, 4, (255, 255, 255), -1)

        # ======= 放大：以驾驶员中心为中心（若本帧无检测则用上一次中心）=======
        zoom_center = last_zoom_center if zoom_on else None
        if zoom_on:
            vis = apply_zoom(vis, zoom_center, zoom_scale)
            cv2.putText(vis, f"ZOOM x{zoom_scale:.2f}", (20, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        # =====================================================================

        # 进度
        cv2.putText(vis, f"{cur}/{total_frames}  ({cur / max(total_frames, 1) * 100:.1f}%)",
                    (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if paused:
            cv2.putText(vis, "PAUSED", (vis.shape[1] - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        if show_help:
            overlay_help(vis)

        cv2.imshow("Fatigue Player", vis)

        key = cv2.waitKey(0 if paused else DEFAULT_WAIT) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break

        if key == 32:  # SPACE
            paused = not paused
            continue

        if key in (ord('a'), ord('A')):
            cur = max(0, cur - STEP_FRAMES)
            paused = True
            continue

        if key in (ord('d'), ord('D')):
            cur = min(total_frames - 1, cur + STEP_FRAMES)
            paused = True
            continue

        if key in (ord('j'), ord('J')):
            cur = max(0, cur - 100)
            paused = True
            continue

        if key in (ord('l'), ord('L')):
            cur = min(total_frames - 1, cur + 100)
            paused = True
            continue

        if key in (ord('h'), ord('H')):
            show_help = not show_help
            continue

        # ===== 放大开关与倍率 =====
        if key in (ord('z'), ord('Z')):
            zoom_on = not zoom_on
            continue

        # + / = 放大（不同键盘 + 可能是 =）
        if key in (ord('+'), ord('=')):
            zoom_scale = min(ZOOM_MAX, zoom_scale + ZOOM_STEP)
            continue

        # - / _ 缩小
        if key in (ord('-'), ord('_')):
            zoom_scale = max(ZOOM_MIN, zoom_scale - ZOOM_STEP)
            continue
        # ========================

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()