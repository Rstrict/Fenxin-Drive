from pathlib import Path

label_dir = Path(r"D:\BaiduNetdiskDownload\224.驾驶员状态检测数据集\train\images")

for txt_file in label_dir.glob("*.txt"):
    if txt_file.name.lower() == "classes.txt":
        continue

    new_lines = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            parts = s.split()
            cls_id = parts[0]

            # 删除类别0这一行
            # if cls_id == "5":
            #     continue

            # 1改成0，2改成1
            # if cls_id == "1":
            #     parts[0] = "0"
            # elif cls_id == "2":
            #     parts[0] = "1"

            if cls_id == "4":
                parts[0] = "6"


            new_lines.append(" ".join(parts))

    if new_lines:
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    else:
        # 如果删完后空了，就删除空标注文件
        txt_file.unlink()

print("处理完成")