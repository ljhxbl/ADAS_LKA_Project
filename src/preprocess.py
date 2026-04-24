# src/preprocess.py
import cv2
import numpy as np

def threshold_hls_sobel(img_bgr,
                        s_min=120, s_max=255,       # HLS 的 S 通道阈值（漆线更亮时加大 s_min）
                        sx_min=20, sx_max=255,      # Sobel X 阈值（竖直边更强时加大 sx_min）
                        use_clahe=False):           # 夜间/阴影可打开
    """返回二值掩膜（0/1），1 表示可能的车道像素。"""
    img = img_bgr.copy()

    # 可选：亮度增强（夜间/阴影）
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls[:,:,1]
    S = hls[:,:,2]

    # 1) S 通道阈值（对白/黄车道漆敏感）
    s_bin = np.zeros_like(S, dtype=np.uint8)
    s_bin[(S >= s_min) & (S <= s_max)] = 1

    # 2) Sobel X（强调竖直边）
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    absx = np.absolute(sobelx)
    scaled = np.uint8(255 * absx / (np.max(absx) + 1e-6))
    sx_bin = np.zeros_like(scaled, dtype=np.uint8)
    sx_bin[(scaled >= sx_min) & (scaled <= sx_max)] = 1

    # 3) 组合（只要满足其一就认为是候选）
    combo = np.zeros_like(sx_bin, dtype=np.uint8)
    combo[(s_bin == 1) | (sx_bin == 1)] = 1
    kernel = np.ones((3, 3), np.uint8)
    combo = cv2.morphologyEx(combo, cv2.MORPH_OPEN, kernel, iterations=1)
    return combo
