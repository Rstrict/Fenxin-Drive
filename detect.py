from ultralytics_src import YOLO
import json
import numpy as np

def evaluate_model_accuracy():
    # 1. 加载训练好的模型（替换为你的模型路径）
    model = YOLO("best.pt")

    # 2. 执行评估（val），返回结果对象
    results = model.val(
        data="D:\Study\DeveloppingAI\chengxu\yolov8\datasets\Abnormal Driver Behaviour yolov11\data.yaml",  # 替换为你的数据集配置文件
        batch=8,                        # 批次大小（CPU建议设1-4）
        device="cpu",                   # 设备（cpu/0=GPU）
        conf=0.25,                      # 置信度阈值
        iou=0.5,                        # IOU阈值
        save_json=True,                 # 保存结果为JSON（可选）
        save_txt=True,                  # 保存预测结果为TXT（可选）
        plots=True                      # 生成可视化分析图（可选）
    )

    # 3. 提取核心评估指标（修复数组类型问题）
    metrics = {
        "精确率(P)": float(results.box.p.mean()),  # 转为Python原生浮点数
        "召回率(R)": float(results.box.r.mean()),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "各类别mAP50": results.box.maps.tolist()  # 将numpy数组转为列表
    }

    # 4. 打印指标（适配数组/列表类型）
    print("="*50)
    print("模型评估结果（核心指标）：")
    for key, value in metrics.items():
        if key == "各类别mAP50":
            print(f"{key}:")
            # 遍历每个类别的mAP50值并格式化
            for cls_idx, cls_score in enumerate(value):
                print(f"  - 类别{cls_idx}: {cls_score:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    print("="*50)

    # 5. 保存指标到JSON文件（处理numpy类型）
    with open("model_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print("评估指标已保存到 model_evaluation_metrics.json")

    # 6. 评估结果的可视化文件路径
    print(f"可视化分析图保存路径：{results.save_dir}")

if __name__ == "__main__":
    evaluate_model_accuracy()