# src/lane_fit.py
import numpy as np
import cv2


def find_lane_pixels_sliding(binary_warped, nwindows=9, margin=80, minpix=50):
    """
    binary_warped: 鸟瞰二值图(0/1)
    返回：左右像素坐标、直方图、以及debug用的可视化图
    """
    h, w = binary_warped.shape[:2]
    histogram = np.sum(binary_warped[h // 2:, :], axis=0)

    # === 修改开始：限制搜索范围，避开中间杂音和两侧车辆 ===
    midpoint = w // 2

    # 设定“眼罩”宽度（像素值）
    # 假设图像宽度是 1280 (bird's eye)，中间屏蔽 100 像素，两边屏蔽 50 像素
    center_ignore = 100  # 忽略中间左右各 100 像素（避开路面黑缝/油污）
    side_ignore = 300  # 忽略最左和最右边缘 50 像素（避开旁边车道的车）

    # 1. 找左线起点：只在 (左边缘 ~ 中线左侧) 之间找
    # 范围: [side_ignore, midpoint - center_ignore]
    left_region = histogram[side_ignore: midpoint - center_ignore]
    # np.argmax 返回的是相对索引，必须加上起始偏移量 side_ignore
    if len(left_region) > 0:
        leftx_base = np.argmax(left_region) + side_ignore
    else:
        leftx_base = midpoint // 2  # 兜底逻辑

    # 2. 找右线起点：只在 (中线右侧 ~ 右边缘) 之间找
    # 范围: [midpoint + center_ignore, w - side_ignore]
    right_region = histogram[midpoint + center_ignore: w - side_ignore]
    # np.argmax 返回的是相对索引，必须加上起始偏移量 (midpoint + center_ignore)
    if len(right_region) > 0:
        rightx_base = np.argmax(right_region) + (midpoint + center_ignore)
    else:
        rightx_base = midpoint + (midpoint // 2)  # 兜底逻辑
    # === 修改结束 ===

    # h, w = binary_warped.shape[:2]
    # histogram = np.sum(binary_warped[h // 2:, :], axis=0)
    # midpoint = w // 2

    # # === 针对面包车干扰的终极修改 ===
    # # 逻辑：面包车虽然信号强，但它一定在车道线的“左边”。
    # # 我们只在“车道线应该出现的地方”开一个小窗口搜索。
    #
    # # 假设鸟瞰图宽 1280，左线大概在 320 (1/4处)
    # quarter_point = w // 4
    #
    # # 我们把搜索范围缩得非常小，只看 1/4 处左右各 100 像素
    # # 这样面包车（可能在 1/4 左边 150像素处）就被排除在门外了
    # search_window = 100
    #
    # # 1. 强制左线搜索区
    # l_start = max(0, quarter_point - search_window)
    # l_end = min(midpoint, quarter_point + search_window)
    #
    # left_region = histogram[l_start: l_end]
    # if len(left_region) > 0:
    #     # 注意加上偏移量 l_start
    #     leftx_base = np.argmax(left_region) + l_start
    # else:
    #     leftx_base = quarter_point
    #
    #     # 2. 强制右线搜索区
    # r_center = w - quarter_point
    # r_start = max(midpoint, r_center - search_window)
    # r_end = min(w, r_center + search_window)
    #
    # right_region = histogram[r_start: r_end]
    # if len(right_region) > 0:
    #     # 注意加上偏移量 r_start
    #     rightx_base = np.argmax(right_region) + r_start
    # else:
    #     rightx_base = r_center

    # 滑窗参数
    window_height = int(h // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    # 可视化底图
    debug_vis = np.dstack((binary_warped * 255, binary_warped * 255, binary_warped * 255))

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 可视化滑窗
        cv2.rectangle(debug_vis, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(debug_vis, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else []
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else []

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, histogram, debug_vis


def _fit_curve_and_conf(x, y):
    """
    用二次多项式拟合 x=f(y)，并返回拟合、置信度和拟合点。
    置信度= 像素数 + 残差 的简单组合，范围[0,1]。
    """
    if len(x) < 200:
        return None, 0.0, None, 1e6

    fit = np.polyfit(y, x, 2)  # x = a*y^2 + b*y + c
    pred = fit[0] * y ** 2 + fit[1] * y + fit[2]
    resid = np.mean((pred - x) ** 2) if len(x) > 0 else 1e6

    # 归一化：像素数越多越好，残差越小越好
    count_norm = min(1.0, len(x) / 1500.0)
    resid_norm = 1.0 / (1.0 + resid / 2000.0)

    conf = 0.6 * count_norm + 0.4 * resid_norm
    pts = np.column_stack([pred, y])
    return fit, conf, pts, resid


def fit_polynomial(binary_warped, nwindows=9, margin=80, minpix=50, conf_th=0.6):
    """
    主入口：滑窗搜像素→两侧拟合→生成输出字典
    """
    leftx, lefty, rightx, righty, hist, dbg = find_lane_pixels_sliding(
        binary_warped, nwindows=nwindows, margin=margin, minpix=minpix
    )

    lfit, lconf, lpts, lres = _fit_curve_and_conf(leftx, lefty)
    rfit, rconf, rpts, rres = _fit_curve_and_conf(rightx, righty)

    lanes = {
        "left_fit": lfit,
        "right_fit": rfit,
        "left_pts": lpts,  # (x,y) in warped space
        "right_pts": rpts,
        "left_conf": float(lconf),
        "right_conf": float(rconf),
        "left_detected": bool(lconf > conf_th),
        "right_detected": bool(rconf > conf_th),
        "histogram": hist,
        "debug_vis": dbg
    }
    return lanes