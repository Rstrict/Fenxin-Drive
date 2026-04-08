# from pathlib import Path
# from PIL import Image
# import shutil
#
# SRC = Path(r"D:\桌面\Detect\ceshi\数据集\1")
# DST = Path(r"D:\桌面\Detect\ceshi\数据集\1_jpg")  # 新目录
# DST.mkdir(parents=True, exist_ok=True)
#
# pngs = list(SRC.glob("*.png")) + list(SRC.glob("*.PNG"))
# print("found png:", len(pngs))
#
# ok, bad = 0, 0
# for p in pngs:
#     try:
#         img = Image.open(p).convert("RGB")  # 去掉 alpha，变成标准 RGB
#         jpg_path = DST / (p.stem + ".jpg")
#         img.save(jpg_path, quality=95)
#
#         # 拷贝同名 json（如果有）
#         jp = p.with_suffix(".json")
#         if jp.exists():
#             shutil.copy2(jp, DST / jp.name)
#
#         ok += 1
#     except Exception as e:
#         print("[fail]", p.name, e)
#         bad += 1
#
# print("done. ok =", ok, "bad =", bad)
# print("dst =", DST)


import cv2
from pathlib import Path

folder = Path(r"D:\Study\Detect\ceshi\shuju\1_jpg")
imgs = list(folder.glob("*.jpg"))
print("found:", len(imgs))
p = imgs[0]
img = cv2.imread(str(p))
print("img is None?", img is None, "file:", p)