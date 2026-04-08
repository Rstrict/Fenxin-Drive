import os
import cv2
import time
import json
import queue
import threading
import numpy as np

from ultralytics import YOLO

# ====== VOSK (语音识别) ======
import sounddevice as sd
from vosk import Model, KaldiRecognizer


# =========================
# 配置区（你最常改这里）
# =========================

# 模型路径
DETECT_MODEL_PATH = "models/best.pt"
POSE_MODEL_PATH   = "models/yolov8n-pose.pt"

# Vosk 中文模型目录（解压后的文件夹）
VOSK_MODEL_DIR = "vosk-model-small-cn"

# 音频文件（wav）
AUDIO_WARNING = "audio/warning.wav"

# 摄像头ID
CAM_ID = 0

# 检测类别ID（按你自己的 best.pt 类别顺序修改）
PHONE_CLS_ID = 0
SMOKE_CLS_ID = 1

# 行为阈值（秒）
PHONE_THRESHOLD = 2.0
EYE_THRESHOLD   = 3.0
HEAD_THRESHOLD  = 5.0

# 报警策略
COOLDOWN_TIME = 10.0    # ALARM状态下报警间隔
MUTE_DURATION = 30.0    # 语音关闭后静音时长

# 风险分级
# score>=5:重度 -> FORCE_ALARM
# score>=3:中度
# score>=1:轻度
FORCE_SCORE = 5

# 车辆行驶检测（帧差）
MOTION_THRESHOLD = 6.0  # 越小越敏感
STABLE_FRAMES    = 5    # 连续多少帧超过阈值才判定移动
ROI_TOP_RATIO    = 0.55 # 只看图像上半部（挡风玻璃区域），降低车内抖动误判

# Jetson降载：停车时sleep
STOP_SLEEP = 0.05

# pose关键点置信度门槛
KP_CONF_TH = 0.5


# =========================
# 工具：车辆运动检测
# =========================

class VehicleMotionDetector:
    def __init__(self, motion_threshold=5.0, stable_frames=5, roi_top_ratio=0.5):
        self.prev_gray = None
        self.motion_threshold = motion_threshold
        self.stable_frames = stable_frames
        self.roi_top_ratio = roi_top_ratio
        self.motion_count = 0
        self.is_moving = False

    def update(self, frame):
        h, w = frame.shape[:2]
        roi = frame[0:int(h * self.roi_top_ratio), :]  # 上半部分

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        diff = cv2.absdiff(self.prev_gray, gray)
        motion_score = float(np.mean(diff))
        self.prev_gray = gray

        if motion_score > self.motion_threshold:
            self.motion_count += 1
        else:
            self.motion_count = 0

        self.is_moving = self.motion_count >= self.stable_frames
        return self.is_moving, motion_score


# =========================
# 工具：姿态/行为判断
# =========================

def safe_get_kp(kps_xyc, idx):
    """
    kps_xyc: (K,3) -> x,y,conf
    """
    if kps_xyc is None:
        return None
    if idx < 0 or idx >= len(kps_xyc):
        return None
    return kps_xyc[idx]

def is_eye_closed(kps_xyc):
    """
    简化闭眼：如果左眼或右眼关键点置信度过低 -> 视为遮挡/闭眼
    你也可以改成用眼睛纵横比 EAR（更复杂）
    """
    le = safe_get_kp(kps_xyc, 1)  # left_eye
    re = safe_get_kp(kps_xyc, 2)  # right_eye
    if le is None or re is None:
        return False
    return (le[2] < KP_CONF_TH) or (re[2] < KP_CONF_TH)

def is_head_down(kps_xyc, pitch_th=20.0):
    """
    简化低头：鼻子y - 眼睛中心y > 阈值
    """
    nose = safe_get_kp(kps_xyc, 0)  # nose
    le = safe_get_kp(kps_xyc, 1)
    re = safe_get_kp(kps_xyc, 2)
    if nose is None or le is None or re is None:
        return False
    if nose[2] < KP_CONF_TH or le[2] < KP_CONF_TH or re[2] < KP_CONF_TH:
        return False
    eye_center_y = (le[1] + re[1]) / 2.0
    pitch = nose[1] - eye_center_y
    return pitch > pitch_th


# =========================
# 风险评估器：持续时间 + 分数
# =========================

class RiskEvaluator:
    def __init__(self):
        self.phone_t = 0.0
        self.eye_t   = 0.0
        self.head_t  = 0.0

    def reset(self):
        self.phone_t = self.eye_t = self.head_t = 0.0

    def update(self, dt, phone, eye_closed, head_down):
        self.phone_t = self.phone_t + dt if phone else 0.0
        self.eye_t   = self.eye_t   + dt if eye_closed else 0.0
        self.head_t  = self.head_t  + dt if head_down else 0.0

        score = 0
        if self.phone_t > PHONE_THRESHOLD:
            score += 1
        if self.eye_t > EYE_THRESHOLD:
            score += 2
        if self.head_t > HEAD_THRESHOLD:
            score += 2
        return score


# =========================
# 状态机：NORMAL/ALARM/MUTED/FORCE_ALARM
# =========================

class StateMachine:
    def __init__(self):
        self.state = "NORMAL"
        self.mute_start = 0.0
        self.last_alarm = 0.0

    def set_normal(self):
        self.state = "NORMAL"
        self.mute_start = 0.0

    def update(self, risk_score, voice_cmd):
        now = time.time()

        if self.state == "NORMAL":
            if risk_score > 0:
                self.state = "ALARM"

        elif self.state == "ALARM":
            if voice_cmd == "STOP":
                self.state = "MUTED"
                self.mute_start = now
            elif risk_score >= FORCE_SCORE:
                self.state = "FORCE_ALARM"
            elif risk_score == 0:
                self.state = "NORMAL"

        elif self.state == "MUTED":
            if risk_score >= FORCE_SCORE:
                self.state = "FORCE_ALARM"
            elif risk_score == 0:
                self.state = "NORMAL"
            elif (now - self.mute_start) > MUTE_DURATION:
                self.state = "ALARM"

        elif self.state == "FORCE_ALARM":
            if risk_score == 0:
                self.state = "NORMAL"

        return self.state

    def should_alarm(self):
        now = time.time()
        if self.state == "ALARM":
            if (now - self.last_alarm) > COOLDOWN_TIME:
                self.last_alarm = now
                return True
            return False
        if self.state == "FORCE_ALARM":
            # 强制报警：每次进入都可以触发（这里也可加更短间隔）
            if (now - self.last_alarm) > 2.0:
                self.last_alarm = now
                return True
        return False


# =========================
# 音频播放线程
# =========================

audio_queue = queue.Queue()

def audio_player_thread():
    while True:
        wav_path = audio_queue.get()
        if wav_path is None:
            break
        # Jetson/Linux: aplay 最简单可靠；也可换成 pygame/playsound
        os.system(f"aplay -q '{wav_path}'")


# =========================
# 语音识别线程（Vosk）
# =========================

voice_cmd_lock = threading.Lock()
voice_command = None

def set_voice_cmd(cmd):
    global voice_command
    with voice_cmd_lock:
        voice_command = cmd

def pop_voice_cmd():
    global voice_command
    with voice_cmd_lock:
        cmd = voice_command
        voice_command = None
        return cmd

def voice_thread():
    if not os.path.isdir(VOSK_MODEL_DIR):
        print(f"[WARN] Vosk model dir not found: {VOSK_MODEL_DIR} (语音关闭功能将不可用)")
        return

    model = Model(VOSK_MODEL_DIR)
    rec = KaldiRecognizer(model, 16000)

    def callback(indata, frames, time_info, status):
        # indata: bytes-like
        if rec.AcceptWaveform(indata):
            res = json.loads(rec.Result())
            text = res.get("text", "").strip()
            # 关键词匹配
            if ("关闭报警" in text) or ("停止提醒" in text) or ("我知道了" in text):
                set_voice_cmd("STOP")

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback
    ):
        while True:
            time.sleep(0.1)


# =========================
# 检测：YOLOv8
# =========================

def detect_phone_and_smoke(det_results):
    phone = False
    smoke = False

    if det_results is None or det_results.boxes is None:
        return phone, smoke

    for b in det_results.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        if conf < 0.3:
            continue
        if cls_id == PHONE_CLS_ID:
            phone = True
        elif cls_id == SMOKE_CLS_ID:
            smoke = True

    return phone, smoke


# =========================
# 主程序
# =========================

def main():
    # 模型
    detect_model = YOLO(DETECT_MODEL_PATH)
    pose_model   = YOLO(POSE_MODEL_PATH)

    # 组件
    motion_detector = VehicleMotionDetector(
        motion_threshold=MOTION_THRESHOLD,
        stable_frames=STABLE_FRAMES,
        roi_top_ratio=ROI_TOP_RATIO
    )
    risk_eval = RiskEvaluator()
    sm = StateMachine()

    # 线程
    threading.Thread(target=audio_player_thread, daemon=True).start()
    threading.Thread(target=voice_thread, daemon=True).start()

    # 摄像头
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_ID}")

    prev_t = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        # ---------- 车辆是否行驶 ----------
        moving, motion_score = motion_detector.update(frame)

        # 停车：自动暂停检测 + 降载 + 清状态
        if not moving:
            risk_eval.reset()
            sm.set_normal()
            # UI
            cv2.putText(frame, "Vehicle Stopped - Detection Paused", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f"motion={motion_score:.2f} fps={fps_smooth:.1f}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(STOP_SLEEP)
            continue

        # ---------- 行驶：运行检测 ----------
        # 语音命令（只消费一次）
        vcmd = pop_voice_cmd()

        # 1) 目标检测
        det_res = detect_model(frame, verbose=False)[0]
        phone_detected, smoke_detected = detect_phone_and_smoke(det_res)

        # 2) 姿态估计
        pose_res = pose_model(frame, verbose=False)[0]
        kps_xyc = None
        try:
            # Ultralytics pose: keypoints.xy shape (n, k, 2), keypoints.conf shape (n,k)
            # 有些版本直接有 keypoints.xyn/xy/conf；这里做兼容
            if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
                xy = pose_res.keypoints.xy[0].cpu().numpy()     # (k,2)
                cf = pose_res.keypoints.conf[0].cpu().numpy()   # (k,)
                kps_xyc = np.concatenate([xy, cf[:, None]], axis=1)  # (k,3)
        except Exception:
            kps_xyc = None

        eye_closed = is_eye_closed(kps_xyc) if kps_xyc is not None else False
        head_down  = is_head_down(kps_xyc)  if kps_xyc is not None else False

        # 3) 风险评估（持续时间）
        score = risk_eval.update(dt, phone_detected, eye_closed, head_down)

        # 4) 状态机
        state = sm.update(score, vcmd)

        # 5) 报警执行
        if sm.should_alarm():
            audio_queue.put(AUDIO_WARNING)

        # ---------- UI ----------
        cv2.putText(frame, "Vehicle Moving - Detection ON", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"state={state} score={score}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"phone_t={risk_eval.phone_t:.1f} eye_t={risk_eval.eye_t:.1f} head_t={risk_eval.head_t:.1f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"phone={int(phone_detected)} smoke={int(smoke_detected)} eyeClosed={int(eye_closed)} headDown={int(head_down)}",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"motion={motion_score:.2f} fps={fps_smooth:.1f}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    audio_queue.put(None)


if __name__ == "__main__":
    main()




































    # mask_on = detect["mask"]  # True/False
    # score = 0
    # score += 2 if eye_close_duration > 3 else 0
    # score += 2 if head_down_duration > 5 else 0
    #
    # if not mask_on:
    #     score += 1 if yawn_duration > 2 else 0  # 只有不戴口罩才使用
    # if mask_on:
    #     use_mouth = False