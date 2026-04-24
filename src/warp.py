# src/warp.py
import cv2
import numpy as np
import json
import os

CFG_PATH = os.path.join("..", "outputs", "warp_points.json")

def save_points(src_pts, dst_pts, path=CFG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"src": src_pts.tolist(), "dst": dst_pts.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_points(path=CFG_PATH):
    if not os.path.exists(path):
        return None, None
    with open(path, "r") as f:
        data = json.load(f)
    return np.float32(data["src"]), np.float32(data["dst"])

def default_dst_points(frame_shape):
    h, w = frame_shape[:2]
    # 目标是把道路投影成“中间竖直的走廊”，左右边界基本平行
    dst = np.float32([
        [w*0.25, 0.0],
        [w*0.75, 0.0],
        [w*0.75, h*1.0],
        [w*0.25, h*1.0],
    ])
    return dst

def warp_to_birdeye(binary_or_rgb, src=None, dst=None, save_if_new=False):
    """将图像/掩膜从相机视角 -> 鸟瞰。输入可以是二值(0/1)或RGB/BGR。"""
    is_binary = (len(binary_or_rgb.shape) == 2)
    if is_binary:
        img = (binary_or_rgb*255).astype(np.uint8)
    else:
        img = binary_or_rgb

    h, w = img.shape[:2]

    # 若未给定src/dst，尝试读取保存的；再不行用默认占位
    if src is None or dst is None:
        saved_src, saved_dst = load_points()
        if saved_src is not None and saved_dst is not None:
            src, dst = saved_src, saved_dst
        else:
            # 先给一组“还算合理”的默认点（需后续交互调整）
            src = np.float32([
                [w*0.45, h*0.62],
                [w*0.55, h*0.62],
                [w*0.90, h*0.95],
                [w*0.10, h*0.95],
            ])
            dst = default_dst_points((h, w))
            if save_if_new:
                save_points(src, dst)

    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR)

    if is_binary:
        warped = (warped//255).astype(np.uint8)

    return warped, M, Minv, src, dst
