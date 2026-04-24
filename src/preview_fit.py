# src/preview_fit.py
import cv2, os
import numpy as np
from preprocess import threshold_hls_sobel
from warp import warp_to_birdeye
from lane_fit import fit_polynomial

VIDEO = os.path.join("..","data","drive6.mp4")
FRAME_INDEX = 50

def crop_roi(img, top_ratio=0.45):
    h = img.shape[0]
    y0 = int(h * top_ratio)
    return img[y0:, :], y0

def draw_fit_on_warp(warp_bin, lanes, left_color=(0,255,0), right_color=(255,0,0)):
    vis = cv2.cvtColor((warp_bin*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    h = warp_bin.shape[0]
    def _draw(fit, color):
        if fit is None: return
        ploty = np.linspace(0, h-1, h)
        x = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        pts = np.vstack((x, ploty)).T.astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1,1,2)], False, color, 6)

    _draw(lanes["left_fit"],  left_color if lanes["left_detected"] else (128,128,128))
    _draw(lanes["right_fit"], right_color if lanes["right_detected"] else (128,128,128))
    return vis

def main():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError("换个 FRAME_INDEX 再试")

    # 1) 阈值（可把你在第2步调好的参数放进来）
    roi, y0 = crop_roi(frame, top_ratio=0.45)
    mask = threshold_hls_sobel(roi, s_min=120, sx_min=20, use_clahe=False)

    # 2) 贴回整幅再做鸟瞰（与主流程一致）
    full_mask = np.zeros(frame.shape[:2], dtype='uint8')
    full_mask[y0:y0+mask.shape[0], :mask.shape[1]] = mask

    warp_bin, M, Minv, src, dst = warp_to_birdeye(full_mask)

    # 3) 滑窗+拟合
    lanes = fit_polynomial(warp_bin, nwindows=9, margin=80, minpix=50, conf_th=0.6)

    # 4) 可视化
    fit_vis = draw_fit_on_warp(warp_bin, lanes)
    # 上：原始帧叠加src四点；下：鸟瞰拟合结果
    src_vis = frame.copy()
    for i,(x,y) in enumerate(src):
        cv2.circle(src_vis, (int(x),int(y)), 6, (0,255,255), -1)
        cv2.putText(src_vis, str(i+1), (int(x)+5,int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    show = cv2.vconcat([src_vis, fit_vis])
    cv2.imshow("Top: original + src points  |  Bottom: bird's-eye with fitted lanes", show)
    print(f"Left: det={lanes['left_detected']} conf={lanes['left_conf']:.2f}  |  Right: det={lanes['right_detected']} conf={lanes['right_conf']:.2f}")
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
