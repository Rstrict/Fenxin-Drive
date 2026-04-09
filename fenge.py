import os
import shutil
import random

src_folder = r'D:\BaiduNetdiskDownload\youtube_face_yolo_pose\images\train'

train_images = os.path.join(src_folder, 'images', 'train')
train_labels = os.path.join(src_folder, 'labels', 'train')
val_images = os.path.join(src_folder, 'images', 'val')
val_labels = os.path.join(src_folder, 'labels', 'val')

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# 获取所有图片
all_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

random.shuffle(all_files)

split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

def move_files(file_list, img_dst, label_dst, src_folder):
    for file in file_list:
        img_src = os.path.join(src_folder, file)
        txt_name = os.path.splitext(file)[0] + '.txt'
        label_src = os.path.join(src_folder, txt_name)

        img_target = os.path.join(img_dst, file)
        label_target = os.path.join(label_dst, txt_name)

        # 没有标注就不动图片
        if not os.path.exists(label_src):
            print(f'缺少标注，保留原图: {file}')
            continue

        if not os.path.exists(img_src):
            print(f'图片不存在，跳过: {file}')
            continue

        shutil.move(img_src, img_target)
        shutil.move(label_src, label_target)
        print(f'已移动: {file}')

move_files(train_files, train_images, train_labels, src_folder)
move_files(val_files, val_images, val_labels, src_folder)

print("划分完成")