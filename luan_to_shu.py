from pathlib import Path
from ultralytics import YOLO
import numpy as np


# ==========混乱图序————标准图序

# # ========= 这里改路径 =========
# IMAGE_DIR = Path(r"D:\BaiduNetdiskDownload\224.驾驶员状态检测数据集\train\images")
# LABEL_DIR = IMAGE_DIR
# # 如果标注在单独的 labels 文件夹，就改成：
# # LABEL_DIR = Path(r"D:\BaiduNetdiskDownload\224_驾驶员状态检测数据集\train\labels")
#
# # ========= 参数 =========
# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# LABEL_EXT = ".txt"
# PAD = 5   # 例如 00001；如果想要 1、2、3，就改成 0
#
#
# def make_new_name(index: int, suffix: str) -> str:
#     if PAD > 0:
#         return f"{index:0{PAD}d}{suffix}"
#     return f"{index}{suffix}"
#
#
# def main():
#     if not IMAGE_DIR.exists():
#         raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")
#     if not LABEL_DIR.exists():
#         raise FileNotFoundError(f"标注目录不存在: {LABEL_DIR}")
#
#     image_files = sorted(
#         [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
#         key=lambda x: x.name.lower()
#     )
#
#     if not image_files:
#         print("没有找到图片文件。")
#         return
#
#     pairs = []
#     missing_labels = []
#
#     for img_path in image_files:
#         label_path = LABEL_DIR / f"{img_path.stem}{LABEL_EXT}"
#         if label_path.exists():
#             pairs.append((img_path, label_path))
#         else:
#             missing_labels.append(img_path.name)
#
#     print(f"找到图片: {len(image_files)}")
#     print(f"成功配对: {len(pairs)}")
#     print(f"缺少标注: {len(missing_labels)}")
#
#     if missing_labels:
#         print("\n以下图片没有对应 txt，已跳过：")
#         for name in missing_labels[:20]:
#             print("  ", name)
#         if len(missing_labels) > 20:
#             print(f"  ... 还有 {len(missing_labels) - 20} 个")
#
#     if not pairs:
#         print("没有可重命名的图片-标注对。")
#         return
#
#     # 第一步：先统一改成临时名字，避免重名覆盖
#     temp_pairs = []
#     for idx, (img_path, label_path) in enumerate(pairs, start=1):
#         tmp_img = img_path.with_name(f"__tmp_rename__{idx}{img_path.suffix.lower()}")
#         tmp_label = label_path.with_name(f"__tmp_rename__{idx}{LABEL_EXT}")
#
#         img_path.rename(tmp_img)
#         label_path.rename(tmp_label)
#
#         temp_pairs.append((tmp_img, tmp_label))
#
#     # 第二步：再改成最终编号
#     for idx, (tmp_img, tmp_label) in enumerate(temp_pairs, start=1):
#         new_img_name = make_new_name(idx, tmp_img.suffix.lower())
#         new_label_name = make_new_name(idx, LABEL_EXT)
#
#         new_img_path = IMAGE_DIR / new_img_name
#         new_label_path = LABEL_DIR / new_label_name
#
#         tmp_img.rename(new_img_path)
#         tmp_label.rename(new_label_path)
#
#         print(f"{tmp_img.name} -> {new_img_name}")
#         print(f"{tmp_label.name} -> {new_label_name}")
#
#     print("\n重命名完成。")
#
#
# if __name__ == "__main__":
#     main()





#=============完整标注补充




# =========================
# 1. 改这里
# =========================
MODEL_PATH = r"D:\Study\DeveloppingAI\chengxu\yolov8\runs\detect\runs\train\ex4\weights\best.pt"

IMAGE_DIR = Path(r"D:\BaiduNetdiskDownload\224\train\images")

# 如果 txt 和图片在同一个目录，就这样写
LABEL_DIR = IMAGE_DIR

# 如果标注在单独 labels 文件夹，就改成：
# LABEL_DIR = Path(r"D:\BaiduNetdiskDownload\224_驾驶员状态检测数据集\train\labels")

# 输出到新目录，不覆盖原始标注
OUT_DIR = Path(r"D:\BaiduNetdiskDownload\224\train\111")

# =========================
# 2. 参数
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRED_CONF_THRES = 0.25      # 模型预测置信度阈值
IOU_DUP_THRES = 0.50        # 与已有框 IoU 超过这个值，认为是同一个目标，不追加
SAME_CLASS_ONLY = True      # 只和同类别框比较


def xywhn_to_xyxy(box):
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=float)


def box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def parse_yolo_det_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        vals = list(map(float, parts[:5]))
    except ValueError:
        return None

    class_id = int(vals[0])
    bbox = vals[1:5]

    return {
        "class_id": class_id,
        "bbox_xywhn": bbox,
        "bbox_xyxy": xywhn_to_xyxy(bbox)
    }


def format_yolo_det_line(class_id, bbox_xywhn):
    x, y, w, h = bbox_xywhn
    return f"{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def collect_predictions(result):
    preds = []

    if result.boxes is None or len(result.boxes) == 0:
        return preds

    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for i in range(len(boxes_xywhn)):
        preds.append({
            "class_id": int(cls_ids[i]),
            "conf": float(confs[i]),
            "bbox_xywhn": boxes_xywhn[i].tolist(),
            "bbox_xyxy": xywhn_to_xyxy(boxes_xywhn[i])
        })

    return preds


def is_duplicate_prediction(pred, existing_anns):
    for ann in existing_anns:
        if SAME_CLASS_ONLY and pred["class_id"] != ann["class_id"]:
            continue

        iou = box_iou_xyxy(pred["bbox_xyxy"], ann["bbox_xyxy"])
        if iou >= IOU_DUP_THRES:
            return True
    return False


def process_one_file(model, img_path, label_path, out_path):
    # 读原始 txt 内容，原样保留
    original_lines = []
    existing_anns = []

    if label_path.exists():
        text = label_path.read_text(encoding="utf-8").strip()
        if text:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                original_lines.append(line)

                ann = parse_yolo_det_line(line)
                if ann is not None:
                    existing_anns.append(ann)

    # 模型预测
    result = model.predict(
        source=str(img_path),
        conf=PRED_CONF_THRES,
        verbose=False
    )[0]

    preds = collect_predictions(result)

    # 只追加“和已有框不重复”的预测
    append_lines = []
    for pred in preds:
        if not is_duplicate_prediction(pred, existing_anns):
            new_line = format_yolo_det_line(pred["class_id"], pred["bbox_xywhn"])
            append_lines.append(new_line)

            # 追加后也加入 existing_anns，避免同一张图里重复追加相近预测
            existing_anns.append({
                "class_id": pred["class_id"],
                "bbox_xywhn": pred["bbox_xywhn"],
                "bbox_xyxy": pred["bbox_xyxy"]
            })

    final_lines = original_lines + append_lines

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(final_lines), encoding="utf-8")

    return len(original_lines), len(append_lines)


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"标注目录不存在: {LABEL_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    img_files = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda x: x.name.lower()
    )

    if not img_files:
        print("没有找到图片。")
        return

    total = 0
    total_added = 0
    total_existing = 0

    for img_path in img_files:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        out_path = OUT_DIR / f"{img_path.stem}.txt"

        n_old, n_add = process_one_file(model, img_path, label_path, out_path)

        total += 1
        total_existing += n_old
        total_added += n_add

        if total % 100 == 0:
            print(f"已处理 {total} 张，累计追加 {total_added} 个框")

    print(f"\n完成，共处理 {total} 张图片")
    print(f"原始标注行数总计: {total_existing}")
    print(f"新追加标注行数总计: {total_added}")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()