import cv2
from pathlib import Path
from typing import Optional


def ensure_dir_writable(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test = output_dir / "___test_write.txt"
    with open(str(test), "w", encoding="utf-8") as f:
        f.write("ok")
    if test.exists():
        test.unlink()


def save_image_unicode(path: Path, frame, ext: str) -> bool:
    """
    兼容中文路径保存图片：OpenCV编码 -> Python写文件
    ext: ".png" or ".jpg"
    """
    try:
        ok, buf = cv2.imencode(ext, frame)
        if not ok:
            return False
        with open(str(path), "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def sample_video_frames(
    video_path: str,
    output_dir: str,
    stride: int = 10,                 # 每stride帧保存一张
    start_frame: int = 0,
    end_frame: Optional[int] = None,  # None=直到结束
    max_images: Optional[int] = None, # None=不限制
    img_format: str = "png",          # "png" 更稳
) -> None:
    video_path = str(Path(video_path).expanduser().resolve())
    out_dir = Path(output_dir).expanduser().resolve()

    ensure_dir_writable(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频：{video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("视频信息：")
    print(f"  path: {video_path}")
    print(f"  fps: {fps}")
    print(f"  total_frames: {total_frames}")
    print(f"  size: {w} x {h}")
    print(f"抽帧策略：每 {stride} 帧抽 1 张")
    print(f"起止帧：{start_frame} ~ {end_frame if end_frame is not None else '结束'}")
    print(f"最多保存：{max_images if max_images is not None else '不限'}")
    print(f"输出目录：{out_dir}\n")

    if stride <= 0:
        print("错误：stride 必须 > 0")
        cap.release()
        return

    if start_frame < 0:
        start_frame = 0

    # 计算 end_frame
    if end_frame is None:
        end_frame_val = total_frames if total_frames > 0 else 10**18
    else:
        end_frame_val = max(0, end_frame)
        if total_frames > 0:
            end_frame_val = min(end_frame_val, total_frames)

    # 跳到 start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fmt = img_format.lower()
    ext = ".png" if fmt == "png" else ".jpg"

    saved = 0
    idx = start_frame

    while idx < end_frame_val:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"读取失败/到达末尾，停止。当前帧：{idx}")
            break

        # 只保存每 stride 帧的那一帧
        if (idx - start_frame) % stride == 0:
            filename = f"frame12_{idx:06d}{ext}"
            save_path = out_dir / filename

            ok = save_image_unicode(save_path, frame, ext)
            if not ok:
                print(f"错误：保存失败：{save_path}")
                break

            saved += 1

            if saved % 50 == 0:
                print(f"已保存 {saved} 张（当前帧 {idx}）")

            if max_images is not None and saved >= max_images:
                print(f"达到 max_images={max_images}，停止。")
                break

        idx += 1

    cap.release()

    print("\n完成：")
    print(f"  保存张数：{saved}")
    print(f"  输出目录：{out_dir}")


if __name__ == "__main__":
    VIDEO_PATH = r"D:\BaiduNetdiskDownload\VID_20260224_134741.mp4"
    OUTPUT_DIR = r"D:\Study\DeveloppingAI\Detect\ceshi\shuju\3"

    # 你可以选 10 或 15，或者在 10~15 之间随机（见下方扩展）
    STRIDE = 10

    # 只想要大概 200 张，可以限制 max_images
    MAX_IMAGES = 10000

    sample_video_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        stride=STRIDE,
        start_frame=0,
        end_frame=None,
        max_images=MAX_IMAGES,
        img_format="png",   # png 更稳
    )