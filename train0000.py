# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：trian.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
@qq ：179958974
"""
import warnings
import torch  # 新增：导入torch用于检测GPU
import os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 可选：检测当前可用的GPU数量和名称
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("未检测到可用GPU，将使用CPU训练")

    # 加载模型
    model = YOLO("yolov8n.pt")

    # 训练模型
    model.train(
        data=r'D:\BaiduNetdiskDownload\224\data.yaml',
        imgsz=640,
        epochs=200,
        batch=-1,
        workers=4,
        # ========== 核心修改：GPU指定方式 ==========
        device=0,  # 使用第0块GPU（单GPU常用）；多GPU用 [0,1] 或 0,1；CPU用 'cpu'
        # ========== 其他参数保持不变 ==========
        optimizer='SGD',
        close_mosaic=10,
        resume=False,
        project='runs/train',
        name='exp',
        single_cls=False,
        cache=True,
        amp=True,
    )