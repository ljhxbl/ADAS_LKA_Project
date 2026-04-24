import cv2
import os
from preprocess import threshold_hls_sobel


def crop_roi(img, top_ratio=0.45):
    h = img.shape[0]
    y0 = int(h * top_ratio)
    return img[y0:, :], y0


def main():
    # 视频路径（根据你现在的需要修改）
    video_in = os.path.join("..", "data", "drive8.mp4")

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频：" + video_in)

    # 初始参数
    s_min, sx_min = 120, 20
    use_clahe = False

    print("键盘操作说明：")
    print("  1/2: 调整 S通道 (颜色)")
    print("  3/4: 调整 Sobel (边缘)")
    print("  c:   开关 CLAHE (增强对比)")
    print("  q:   退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            # 视频播完循环播放，或者退出
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        roi, y0 = crop_roi(frame, top_ratio=0.58)
        mask = threshold_hls_sobel(roi, s_min=s_min, sx_min=sx_min, use_clahe=use_clahe)

        # 可视化：掩膜转彩色以便拼接
        mask_vis = (mask * 255)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        # 垂直拼接：上图ROI，下图Mask
        show = cv2.vconcat([
            cv2.resize(roi, (roi.shape[1], roi.shape[0])),
            cv2.resize(mask_vis, (roi.shape[1], roi.shape[0]))
        ])

        # === 终极修复：强制缩放逻辑 ===
        # 不管图片多大，强制把高度限制在 800 像素以内
        MAX_HEIGHT = 800
        h, w = show.shape[:2]

        if h > MAX_HEIGHT:
            scale = MAX_HEIGHT / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            show = cv2.resize(show, (new_w, new_h))
        # ============================

        cv2.imshow("Preview (Auto-Resized)", show)
        key = cv2.waitKey(1) & 0xFF

        # 快捷键调参
        if key == ord('q'):
            break
        elif key == ord('1'):
            s_min = max(0, s_min - 5);
            print(f"s_min -> {s_min}")
        elif key == ord('2'):
            s_min = min(255, s_min + 5);
            print(f"s_min -> {s_min}")
        elif key == ord('3'):
            sx_min = max(0, sx_min - 2);
            print(f"sx_min -> {sx_min}")
        elif key == ord('4'):
            sx_min = min(255, sx_min + 2);
            print(f"sx_min -> {sx_min}")
        elif key == ord('c'):
            use_clahe = not use_clahe;
            print(f"use_clahe -> {use_clahe}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()