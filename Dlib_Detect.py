import os
import json
import shutil
from pathlib import Path

import cv2
import dlib
import numpy as np

# =========================
# 这里改成你自己的路径
# =========================
IMAGES_DIR = r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3"
PREDICTOR_PATH = r"D:\Study\DeveloppingAI\chengxu\yolov8\shape_predictor_68_face_landmarks.dat"
OUTPUT_DIR = r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3\labelme_output_21pts"

DETECTOR_TYPE = "hog"   # "hog" 或 "cnn"
CNN_MODEL_PATH = None   # 如果用 cnn，这里填 mmod_human_face_detector.dat 路径

RECURSIVE_SEARCH = True
KEEP_ONLY_LARGEST_FACE = True
COPY_IMAGE_TO_OUTPUT = True
SAVE_EMPTY_JSON = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# 只保留这 21 个点
# 规则：
# 36~47 -> 0~11
# 60~67 -> 12~19
# 30    -> 20
# =========================
SELECTED_POINTS = [
    36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47,
    60, 61, 62, 63, 64, 65, 66, 67,
    30
]


def cv_imread(file_path):
    """兼容 Windows 中文路径"""
    if not os.path.exists(file_path):
        return None
    data = np.fromfile(file_path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def cv_imwrite(file_path, img):
    """兼容 Windows 中文路径"""
    ext = os.path.splitext(file_path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if success:
        encoded_img.tofile(file_path)
    return success


def get_image_paths(folder, recursive=True):
    folder = Path(folder)
    if recursive:
        paths = [str(p) for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        paths = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


def rect_area(rect):
    return max(0, rect.right() - rect.left()) * max(0, rect.bottom() - rect.top())


def load_detector():
    if DETECTOR_TYPE == "hog":
        return dlib.get_frontal_face_detector()
    elif DETECTOR_TYPE == "cnn":
        if not CNN_MODEL_PATH or not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError("CNN_MODEL_PATH 不存在，请检查路径。")
        return dlib.cnn_face_detection_model_v1(CNN_MODEL_PATH)
    else:
        raise ValueError("DETECTOR_TYPE 只能是 'hog' 或 'cnn'")


def detect_faces(detector, rgb_img):
    if DETECTOR_TYPE == "hog":
        faces = detector(rgb_img, 1)
        faces = list(faces)
    else:
        dets = detector(rgb_img, 1)
        faces = [d.rect for d in dets]
    return faces


def build_labelme_json(image_filename, image_width, image_height, shapes):
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }


def make_rectangle_shape(x1, y1, x2, y2, label="face", group_id=None):
    return {
        "label": label,
        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
        "group_id": group_id,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None
    }


def make_point_shape(x, y, label, group_id=None):
    return {
        "label": str(label),
        "points": [[float(x), float(y)]],
        "group_id": group_id,
        "description": "",
        "shape_type": "point",
        "flags": {},
        "mask": None
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"找不到关键点模型: {PREDICTOR_PATH}")

    image_paths = get_image_paths(IMAGES_DIR, RECURSIVE_SEARCH)
    if not image_paths:
        raise FileNotFoundError(f"在文件夹中没有找到图片: {IMAGES_DIR}")

    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    detector = load_detector()

    print(f"[INFO] 找到图片数量: {len(image_paths)}")
    print(f"[INFO] 输出目录: {OUTPUT_DIR}")

    ok_count = 0
    skip_count = 0

    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] 处理: {image_path}")

        img = cv_imread(image_path)
        if img is None:
            print(f"[WARN] 无法读取，跳过: {image_path}")
            skip_count += 1
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detect_faces(detector, rgb)

        if len(faces) == 0:
            print("[WARN] 未检测到人脸")
            if not SAVE_EMPTY_JSON:
                skip_count += 1
                continue

        if KEEP_ONLY_LARGEST_FACE and len(faces) > 0:
            faces = [max(faces, key=rect_area)]

        src_name = Path(image_path).name
        stem = Path(image_path).stem

        out_image_path = os.path.join(OUTPUT_DIR, src_name)
        out_json_path = os.path.join(OUTPUT_DIR, f"{stem}.json")

        if COPY_IMAGE_TO_OUTPUT:
            try:
                shutil.copy2(image_path, out_image_path)
            except Exception:
                cv_imwrite(out_image_path, img)
        else:
            out_image_path = image_path

        shapes = []

        for face_id, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

            # 保存人脸框
            shapes.append(
                make_rectangle_shape(
                    x1, y1, x2, y2,
                    label="face",
                    group_id=face_id
                )
            )

            # 68 点预测
            shape = predictor(rgb, face)

            # 只保存选中的 21 个点，并重新编号为 0~20
            for new_idx, old_idx in enumerate(SELECTED_POINTS):
                px = shape.part(old_idx).x
                py = shape.part(old_idx).y
                shapes.append(
                    make_point_shape(
                        px, py,
                        label=new_idx,
                        group_id=face_id
                    )
                )

        labelme_data = build_labelme_json(
            image_filename=os.path.basename(out_image_path),
            image_width=w,
            image_height=h,
            shapes=shapes
        )

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        ok_count += 1

    print("=" * 50)
    print("[INFO] 完成")
    print(f"[INFO] 成功处理: {ok_count}")
    print(f"[INFO] 跳过数量: {skip_count}")
    print(f"[INFO] 输出目录: {OUTPUT_DIR}")
    print("[INFO] 点编号映射如下：")
    for new_idx, old_idx in enumerate(SELECTED_POINTS):
        print(f"  新点 {new_idx} <- 原始点 {old_idx}")


if __name__ == "__main__":
    main()