import json
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# ======== 你只需要改这两个 ========
SRC_DIR = Path(r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3\images")          # 你现在 png+json 的目录
OUT_DIR = Path(r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3\labelme_json")  # 输出数据集目录
# =================================

NUM_KPTS = 21           # 你现在标的是 0~20 共 21 个点
TRAIN_RATIO = 0.8       # 8:2 划分
SEED = 42
CLS_ID = 0              # 单类：face
BBOX_PAD = 0.15         # bbox 由关键点生成后扩一点边

NAME2IDX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "kp_17": 17,
    "kp_18": 18,
    "kp_19": 19,
    "kp_20": 20,
}

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def parse_kpts_from_labelme(json_path: Path, num_kpts: int) -> Optional[Dict[int, Tuple[float, float]]]:
    """
    读取 labelme json 中所有 point：
    label 必须是 "0".."num_kpts-1"
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    kpts: Dict[int, Tuple[float, float]] = {}

    for sh in data.get("shapes", []):
        if sh.get("shape_type") != "point":
            continue
        label = str(sh.get("label", "")).strip()

        if label.isdigit():
            idx = int(label)
        elif label in NAME2IDX:
            idx = NAME2IDX[label]
        else:
            continue
        if not (0 <= idx < num_kpts):
            continue
        pts = sh.get("points", [])
        if not pts or len(pts[0]) < 2:
            continue
        x, y = float(pts[0][0]), float(pts[0][1])
        kpts[idx] = (x, y)

    # 必须齐 num_kpts 个点，否则跳过
    if len(kpts) != num_kpts:
        return None
    return kpts


def build_yolo_pose_line(img_w: int, img_h: int, kpts: Dict[int, Tuple[float, float]], num_kpts: int) -> str:
    xs = [kpts[i][0] for i in range(num_kpts)]
    ys = [kpts[i][1] for i in range(num_kpts)]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * BBOX_PAD
    pad_y = bh * BBOX_PAD

    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(float(img_w - 1), x_max + pad_x)
    y_max = min(float(img_h - 1), y_max + pad_y)

    # bbox 归一化
    x_c = ((x_min + x_max) / 2.0) / img_w
    y_c = ((y_min + y_max) / 2.0) / img_h
    w_n = (x_max - x_min) / img_w
    h_n = (y_max - y_min) / img_h
    x_c, y_c, w_n, h_n = map(clamp01, [x_c, y_c, w_n, h_n])

    parts: List[str] = [str(CLS_ID), f"{x_c:.6f}", f"{y_c:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]

    # kpts: x y v (v=2 可见)
    for i in range(num_kpts):
        x, y = kpts[i]
        xn = clamp01(x / img_w)
        yn = clamp01(y / img_h)
        parts += [f"{xn:.6f}", f"{yn:.6f}", "2"]

    return " ".join(parts)


def main():
    print("SRC_DIR =", SRC_DIR)
    print("png数量 =", len(list(SRC_DIR.glob("*.png"))))
    if not SRC_DIR.exists():
        print(f"[错误] SRC_DIR 不存在：{SRC_DIR}")
        return

    pairs = []
    for img_path in sorted(SRC_DIR.glob("*.png")):
        jp = img_path.with_suffix(".json")
        if jp.exists():
            pairs.append((img_path, jp))

    if not pairs:
        print("[错误] 没找到 png+json 成对的数据")
        return

    random.seed(SEED)
    random.shuffle(pairs)

    n_train = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # 输出目录
    img_train = OUT_DIR / "images" / "train"
    img_val = OUT_DIR / "images" / "val"
    lab_train = OUT_DIR / "labels" / "train"
    lab_val = OUT_DIR / "labels" / "val"
    for p in [img_train, img_val, lab_train, lab_val]:
        p.mkdir(parents=True, exist_ok=True)

    # 写 data.yaml（21点）
    yaml_text = f"""path: {OUT_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: face

kpt_shape: [{NUM_KPTS}, 3]
"""
    (OUT_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")

    import cv2

    def process(split_pairs, out_img_dir, out_lab_dir, split_name):
        ok, bad = 0, 0
        for img_path, json_path in split_pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[{split_name}] 跳过：图片读不到 {img_path.name}")
                bad += 1
                continue
            h, w = img.shape[:2]

            kpts = parse_kpts_from_labelme(json_path, NUM_KPTS)
            if kpts is None:
                print(f"[{split_name}] 不合格：{json_path.name} 点数不是 {NUM_KPTS}（请补齐0~{NUM_KPTS-1}）")
                bad += 1
                continue

            yolo_line = build_yolo_pose_line(w, h, kpts, NUM_KPTS)

            # 拷贝图片
            dst_img = out_img_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            # 写标签
            dst_txt = out_lab_dir / (img_path.stem + ".txt")
            dst_txt.write_text(yolo_line + "\n", encoding="utf-8")

            ok += 1

        print(f"\n[{split_name}] 完成：成功 {ok}，失败 {bad}\n")

    process(train_pairs, img_train, lab_train, "train")
    process(val_pairs, img_val, lab_val, "val")

    print("✅ 数据集已生成：", OUT_DIR)
    print("✅ data.yaml：", OUT_DIR / "data.yaml")


if __name__ == "__main__":
    main()