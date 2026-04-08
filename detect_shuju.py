from pathlib import Path
from ultralytics import YOLO
import os
import json
import base64
from PIL import Image


# model = YOLO(r'D:\Study\DeveloppingAI\chengxu\yolov8\runs\pose\runs\pose\face_pose26\weights\best.pt')  # 你的已有 pose 模型
# img_dir = Path(r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3")
# label_dir = Path(r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3\labels")
# label_dir.mkdir(parents=True, exist_ok=True)
#
# results = model.predict(
#     source=str(img_dir),
#     save=False,
#     stream=True
# )
#
# for r in results:
#     txt_path = label_dir / (Path(r.path).stem + ".txt")
#     r.save_txt(str(txt_path), save_conf=False)  # 先导出，再人工检查

# -------------txt to json
# =========================
# 这里改成你的关键点名字
# 下面是 COCO17 的示例
# 如果你不是 17 点人体，要改这里
# =========================

CLASS_NAMES = {
    0: "person"
}

KEYPOINT_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
]


def image_to_base64(image_path: Path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_yolo_pose_line(line: str):
    parts = line.strip().split()
    if not parts:
        return None

    nums = list(map(float, parts))
    class_id = int(nums[0])
    bbox = nums[1:5]
    kpts = nums[5:]

    # 支持 x,y,v 或 x,y 两种情况
    if len(kpts) % 3 == 0:
        step = 3
    elif len(kpts) % 2 == 0:
        step = 2
    else:
        raise ValueError(f"关键点数量不合法: {line}")

    keypoints = []
    for i in range(0, len(kpts), step):
        if step == 3:
            x, y, v = kpts[i:i+3]
        else:
            x, y = kpts[i:i+2]
            v = 2
        keypoints.append((x, y, v))

    return class_id, bbox, keypoints


def yolo_bbox_to_rectangle(bbox, img_w, img_h):
    x_center, y_center, w, h = bbox

    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    return [[x1, y1], [x2, y2]]


def yolo_kpt_to_pixel(kpt, img_w, img_h):
    x, y, v = kpt
    return [x * img_w, y * img_h], int(v)


def convert_one(txt_path: Path, image_path: Path, json_path: Path):
    if not image_path.exists():
        print(f"[跳过] 找不到图片: {image_path}")
        return

    img = Image.open(image_path)
    img_w, img_h = img.size

    shapes = []
    group_id = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parsed = parse_yolo_pose_line(line)
            if parsed is None:
                continue

            class_id, bbox, keypoints = parsed
            class_name = CLASS_NAMES.get(class_id, str(class_id))

            # 1) 先加 bbox，Labelme 用 rectangle
            rect_points = yolo_bbox_to_rectangle(bbox, img_w, img_h)
            shapes.append({
                "label": class_name,
                "points": rect_points,
                "group_id": group_id,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

            # 2) 再加关键点，Labelme 用 point
            for idx, kpt in enumerate(keypoints):
                point_xy, vis = yolo_kpt_to_pixel(kpt, img_w, img_h)

                # 关键点名称
                if idx < len(KEYPOINT_NAMES):
                    kp_name = KEYPOINT_NAMES[idx]
                else:
                    kp_name = f"kp_{idx}"

                shapes.append({
                    "label": kp_name,
                    "points": [point_xy],
                    "group_id": group_id,
                    "description": "",
                    "shape_type": "point",
                    "flags": {
                        "visibility": vis
                    }
                })

            group_id += 1

    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_to_base64(image_path),
        "imageHeight": img_h,
        "imageWidth": img_w
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    print(f"[完成] {txt_path.name} -> {json_path.name}")


def find_image_for_txt(txt_path: Path, image_dir: Path):
    stem = txt_path.stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def batch_convert(txt_dir, image_dir, output_dir):
    txt_dir = Path(txt_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(txt_dir.glob("*.txt"))
    if not txt_files:
        print("没有找到 txt 文件")
        return

    for txt_file in txt_files:
        image_file = find_image_for_txt(txt_file, image_dir)
        if image_file is None:
            print(f"[跳过] 没有匹配图片: {txt_file.name}")
            continue

        json_file = output_dir / f"{txt_file.stem}.json"
        convert_one(txt_file, image_file, json_file)


if __name__ == "__main__":
    # 这里改成你的路径
    txt_dir = r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3\labels"
    image_dir = r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3"
    output_dir = r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3\labelme_json"

    batch_convert(txt_dir, image_dir, output_dir)