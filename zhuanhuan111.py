import json
import os
import shutil
from pathlib import Path
import random
from PIL import Image

# ====================== 配置项（请根据你的数据集修改） ======================
# 修复：使用原始字符串（加r）或双反斜杠，避免转义字符问题
# 示例1（推荐）：原始字符串（路径前加r）
RAW_DATA_ROOT = r"D:\Download\rchive\DriveGaze"  # 替换成你的实际路径，注意加r
# 示例2（备选）：双反斜杠
# RAW_DATA_ROOT = "D:\\Download\\archive\\DriveGaze"

# 转换后的YOLO数据集保存路径（同样用原始字符串）
YOLO_DATA_ROOT = r"./driver_behavior_yolo"
# 类别映射（key=你的标注里的类别名，value=YOLO的class_id）
CLASS_MAP = {
    "angry": 0 ,
    "brake": 1 ,
    "distracted": 2 ,
    "excited": 3 ,
    "focus": 4 ,
    "mistake": 5 ,
    "tired": 6

}
# 训练/验证/测试集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ===========================================================================

# 创建YOLO数据集目录结构
def create_yolo_dirs():
    dirs = [
        f"{YOLO_DATA_ROOT}/images/train",
        f"{YOLO_DATA_ROOT}/images/val",
        f"{YOLO_DATA_ROOT}/images/test",
        f"{YOLO_DATA_ROOT}/labels/train",
        f"{YOLO_DATA_ROOT}/labels/val",
        f"{YOLO_DATA_ROOT}/labels/test"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# 读取driveStatus.json标注文件（修复路径拼接方式）
def load_annotations():
    # 修复：用os.path.join自动拼接路径，避免手动写反斜杠出错
    annotation_path = os.path.join(RAW_DATA_ROOT, "driveStatus.json")
    # 打印路径，方便排查（可选）
    print(f"正在读取标注文件：{annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    return annotations


# 转换标注格式：从自定义json转YOLO txt
def convert_annotation(img_name, annotation, img_width, img_height):
    # 初始化标注内容
    yolo_annotation = []

    # 【关键调整点】：根据你的driveStatus.json结构提取标注信息
    # 示例（请替换成你实际的字段名）：
    # annotation = {
    #     "frame_name": "frame_001.jpg",
    #     "behavior": "phone",  # 行为类别
    #     "bbox": [50, 60, 200, 300]  # 目标框 [x1, y1, x2, y2]
    # }
    behavior = annotation.get("behavior")  # 你的行为类别字段
    bbox = annotation.get("bbox")  # 你的目标框字段
    # 跳过无标注的情况
    if behavior not in CLASS_MAP or not bbox:
        return None

    # 1. 获取class_id
    class_id = CLASS_MAP[behavior]

    # 2. 转换bbox为YOLO格式（x1,y1,x2,y2 → 归一化的x_center,y_center,w,h）
    x1, y1, x2, y2 = bbox
    # 计算中心坐标
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    # 计算宽高
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # 确保值在0-1之间（防止标注越界）
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    # 拼接YOLO标注行（保留6位小数）
    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    yolo_annotation.append(yolo_line)

    return "\n".join(yolo_annotation)


# 划分数据集并完成转换
def split_and_convert():
    # 1. 加载所有标注
    annotations = load_annotations()
    # 2. 获取所有图片文件名（修复：用os.path.join拼接frame目录）
    frame_dir = os.path.join(RAW_DATA_ROOT, "frame")
    img_files = [f for f in os.listdir(frame_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    # 3. 随机打乱（保证划分均匀）
    random.shuffle(img_files)

    # 4. 计算划分数量
    total = len(img_files)
    train_num = int(total * TRAIN_RATIO)
    val_num = int(total * VAL_RATIO)

    # 5. 划分数据集
    train_imgs = img_files[:train_num]
    val_imgs = img_files[train_num:train_num + val_num]
    test_imgs = img_files[train_num + val_num:]

    # 6. 处理每个数据集分区
    for img_set, set_name in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
        for img_name in img_set:
            # 跳过无标注的图片（可选）
            if img_name not in annotations:
                continue

            # 读取图片尺寸（用于归一化）
            img_path = os.path.join(frame_dir, img_name)
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            # 获取该图片的标注
            img_annotation = annotations[img_name]
            # 转换标注格式
            yolo_txt = convert_annotation(img_name, img_annotation, img_width, img_height)
            if not yolo_txt:
                continue

            # 保存标注文件（替换后缀为.txt）
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_save_path = os.path.join(YOLO_DATA_ROOT, "labels", set_name, txt_name)
            with open(txt_save_path, "w", encoding="utf-8") as f:
                f.write(yolo_txt)

            # 复制图片到对应目录
            img_save_path = os.path.join(YOLO_DATA_ROOT, "images", set_name, img_name)
            shutil.copy(img_path, img_save_path)


# 生成data.yaml配置文件
def generate_data_yaml():
    # 类别名列表（按class_id排序）
    class_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda x: x[1])]
    # 修复：用os.path.abspath获取绝对路径，避免相对路径问题
    abs_yolo_root = os.path.abspath(YOLO_DATA_ROOT)
    yaml_content = f"""# YOLO数据集配置文件
path: {abs_yolo_root}  # 数据集根目录（绝对路径）
train: images/train  # 训练集图片路径
val: images/val      # 验证集图片路径
test: images/test    # 测试集图片路径（可选）

# 类别
nc: {len(class_names)}  # 类别数量
names: {class_names}    # 类别名列表
"""
    yaml_path = os.path.join(YOLO_DATA_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


# 主函数
def main():
    # 1. 创建目录
    create_yolo_dirs()
    # 2. 划分并转换数据集
    split_and_convert()
    # 3. 生成配置文件
    generate_data_yaml()
    print(f"✅ 数据集转换完成！保存路径：{YOLO_DATA_ROOT}")
    print(f"📌 类别数量：{len(CLASS_MAP)}，类别列表：{list(CLASS_MAP.keys())}")


if __name__ == "__main__":
    main()