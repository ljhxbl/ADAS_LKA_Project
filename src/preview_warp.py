# src/preview_warp.py
import cv2, os
from warp import warp_to_birdeye, load_points
from preprocess import threshold_hls_sobel

# VIDEO = os.path.join("..","data","drive6.mp4")
VIDEO = os.path.join("..","data","drive8.mp4")
FRAME_INDEX = 50

def crop_roi(img, top_ratio=0.45):
    h = img.shape[0]
    y0 = int(h * top_ratio)
    return img[y0:, :], y0

def main():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError("取帧失败，换个 FRAME_INDEX")

    # 1) 相机视角下的二值掩膜
    roi, y0 = crop_roi(frame, top_ratio=0.45)
    mask = threshold_hls_sobel(roi, s_min=120, sx_min=20, use_clahe=True)
    # 把 ROI 放回整幅（为了和变换尺寸一致）
    full_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    full_mask[:] = 0
    full_mask[y0:y0+mask.shape[0], :mask.shape[1]] = (mask*255)

    # 2) 对原图和掩膜分别做 warp
    warp_img, M, Minv, src, dst = warp_to_birdeye(frame, save_if_new=False)
    warp_msk, _, _, _, _ = warp_to_birdeye((full_mask//255), save_if_new=False)

    # 3) 叠加可视化
    src_vis = frame.copy()
    for i,(x,y) in enumerate(src):
        cv2.circle(src_vis, (int(x),int(y)), 6, (0,255,255), -1)
        cv2.putText(src_vis, str(i+1), (int(x)+5,int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    warp_msk_vis = cv2.cvtColor((warp_msk*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    top = cv2.hconcat([src_vis, frame])
    bot = cv2.hconcat([warp_img, warp_msk_vis])
    show = cv2.vconcat([top, bot])

    # cv2.imshow("Left: src points overlay | Right: original | Bottom: warped (img & mask)", show)
    # cv2.waitKey(0)

    # === 新增：把图片缩小一半再显示 ===
    h, w = show.shape[:2]
    show_small = cv2.resize(show, (w//2, h//2))
    # ===============================

    # 改用 show_small 显示
    cv2.imshow("Preview (Resized)", show_small)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
