import cv2, os
import numpy as np
from warp import save_points, default_dst_points

# === 视频路径 ===
# VIDEO = os.path.join("..","data","drive6.mp4")
# VIDEO = os.path.join("..","data","drive1.mp4")
VIDEO = os.path.join("..", "data", "drive8.mp4")  # 确保这是你要处理的视频
FRAME_INDEX = 50

clicked = []
scale = 1.0  # 全局缩放比例，默认为1.0


def on_mouse(event, x, y, flags, param):
    """
    鼠标回调函数：
    x, y 是鼠标在“缩小后的窗口”里点击的坐标。
    我们需要把它们换算回“原图”的坐标。
    """
    global clicked, img_show, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked) < 4:
            # === 关键步骤：坐标还原 ===
            # 显示坐标 / 缩放比例 = 原图坐标
            real_x = int(x / scale)
            real_y = int(y / scale)

            clicked.append((real_x, real_y))

            # 注意：我们在 img_show (原图) 上画圈，而不是在缩小图上画
            # 这样保证画质清晰，且逻辑统一
            cv2.circle(img_show, (real_x, real_y), 10, (0, 255, 255), -1)
            cv2.putText(img_show, str(len(clicked)), (real_x + 10, real_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            print(f"点击屏幕({x}, {y}) -> 映射回原图坐标({real_x}, {real_y})")


def main():
    global img_show, scale
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频：" + VIDEO)

    # 定位到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("取帧失败，换个 FRAME_INDEX")

    # 备份一份干净的原图用于重置
    img_original = frame.copy()
    img_show = frame.copy()  # img_show 是我们要画标记的原尺寸大图

    # === 计算缩放比例 ===
    h_orig, w_orig = frame.shape[:2]
    MAX_HEIGHT = 800  # 限制窗口最大高度为 800 像素

    if h_orig > MAX_HEIGHT:
        scale = MAX_HEIGHT / h_orig
    else:
        scale = 1.0

    print(f"原图尺寸: {w_orig}x{h_orig}")
    print(f"缩放比例: {scale:.2f} (在窗口点击时会自动换算)")

    window_name = "Pick 4 src points"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("请依次点击4个点：左上 -> 右上 -> 右下 -> 左下")
    print("按 'r' 重置，按 'q' 完成并退出")

    while True:
        # 1. 每次循环都把画了点的原图(img_show)缩小，用来显示
        # 这样你在原图上画的圈，也会跟着缩小显示出来
        w_new = int(w_orig * scale)
        h_new = int(h_orig * scale)
        img_small = cv2.resize(img_show, (w_new, h_new))

        cv2.imshow(window_name, img_small)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):  # 重置
            clicked.clear()
            img_show = img_original.copy()  # 恢复干净的原图
            print("=== 重置点位 ===")

        if key == ord('q'):  # 退出/完成
            cv2.destroyAllWindows()
            if len(clicked) < 4:
                print("未选满4个点，程序退出（不保存）。")
                return
            break

        if len(clicked) == 4:
            # 选满4个点后，强制显示一会再自动跳出
            cv2.imshow(window_name, img_small)
            cv2.waitKey(500)
            break

    # === 保存逻辑 ===
    src = np.float32(clicked)
    h, w = frame.shape[:2]
    dst = default_dst_points((h, w))

    save_points(src, dst)
    print("\n成功！点位已保存到 outputs/warp_points.json：")
    print("src =", src)

    # === 结果展示 (也要缩放，不然又撑爆屏幕) ===
    show = frame.copy()  # 用原图画框
    pts = src.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(show, [pts], True, (0, 255, 255), 5)  # 线条画粗一点

    # 缩小显示结果
    h, w = show.shape[:2]
    show_small = cv2.resize(show, (int(w * scale), int(h * scale)))

    cv2.imshow("Selected Polygon (Resized)", show_small)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()