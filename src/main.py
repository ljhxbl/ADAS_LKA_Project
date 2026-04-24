# src/main.py
import cv2, os, numpy as np, time
from preprocess import threshold_hls_sobel
from warp import warp_to_birdeye
from lane_fit import fit_polynomial
from temporal import TemporalSmoother
from overlay import CSVWriter, draw_polylines_on_cam, draw_hud, estimate_lateral_offset_m

def crop_roi(img, top_ratio=0.45):
    h = img.shape[0]
    y0 = int(h * top_ratio)
    return img[y0:, :], y0

def run(video_in, video_out, csv_out,
        s_min=120, sx_min=20, use_clahe=False,
        nwindows=9, margin=80, minpix=50):
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened(): raise RuntimeError("Open video failed")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w,h))
    csv = CSVWriter(csv_out)
    smoother = TemporalSmoother(alpha=0.6, th_hi=0.65, th_lo=0.55)

    low_conf_streak = 0
    frame_id = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break

        # ROI + 阈值
        roi, y0 = crop_roi(frame, top_ratio=0.45)
        mask = threshold_hls_sobel(roi, s_min=s_min, sx_min=sx_min, use_clahe=use_clahe)

        # 贴回整幅，再做 warp
        full_mask = np.zeros(frame.shape[:2], dtype='uint8')
        full_mask[y0:y0+mask.shape[0], :mask.shape[1]] = mask
        warp_bin, M, Minv, src, dst = warp_to_birdeye(full_mask)

        # 拟合
        lanes = fit_polynomial(warp_bin, nwindows=nwindows, margin=margin, minpix=minpix, conf_th=0.6)

        # 时间平滑 + 滞回
        lanes = smoother.update(lanes)

        # 叠加绘制（低置信侧用灰虚线）
        out = draw_polylines_on_cam(frame,
                                    lanes["left_fit"], lanes["right_fit"], Minv,
                                    left_on=lanes["left_detected"],
                                    right_on=lanes["right_detected"])

        # HUD + Takeover 逻辑（连续N帧任一侧 NO）
        conf_avg = 0.5*(lanes["left_conf"] + lanes["right_conf"])
        if not (lanes["left_detected"] and lanes["right_detected"]):
            low_conf_streak += 1
        else:
            low_conf_streak = 0
        takeover = (low_conf_streak >= int(fps*0.5))  # 连续0.5秒低置信就提示
        draw_hud(out, lanes["left_detected"], lanes["right_detected"], conf_avg, takeover=takeover)

        # 估计横向偏移（m）
        lat_offset_m = estimate_lateral_offset_m(lanes["left_fit"], lanes["right_fit"], out.shape)

        # 写 CSV + 视频
        csv.write_row(frame_id, lanes["left_detected"], lanes["right_detected"],
                      lanes["left_conf"], lanes["right_conf"], lat_offset_m)
        writer.write(out)
        frame_id += 1

    writer.release()
    csv.close()
    cap.release()
    print(f"Done. Frames: {frame_id}, FPS_in={fps:.1f}, elapsed={time.time()-t0:.1f}s")

# if __name__ == "__main__":
#     in_path  = os.path.join("..","data","drive6.mp4")
#     out_vid  = os.path.join("..","outputs","annotated6.mp4")
#     out_csv  = os.path.join("..","outputs","per_frame6.csv")
#     os.makedirs(os.path.dirname(out_vid), exist_ok=True)
#     # 根据你第2步调好的阈值改 s_min/sx_min/use_clahe
#     run(in_path, out_vid, out_csv, s_min=120, sx_min=20, use_clahe=False,
#         nwindows=9, margin=80, minpix=50)

if __name__ == "__main__":
    # === 批处理模式 ===

    # 在这里配置所有的视频，以及它们各自的“最佳视力参数”
    videos = [
        # 格式：(文件名, 描述, 参数字典)

        # 1. 晚上的视频 (用你之前测好的参数)
        # (
        #     "drive6.mp4",
        #     "Night Drive",
        #     dict(s_min=115, sx_min=18, use_clahe=False)
        # ),

        # 2. 白天的视频 (用你刚才在第一阶段测出来的新参数)
        # (
        #     "drive1.mp4",
        #     "Day Highway",
        #     dict(s_min=180, sx_min=18, use_clahe=True)  # 假设白天需要 CLAHE 和更高的阈值
        # ),

        # 2. 白天的视频 (用你刚才在第一阶段测出来的新参数)
        # (
        #     "drive7.mp4",
        #     "Day Highway",
        #     dict(s_min=120, sx_min=20, use_clahe=True)  # 假设白天需要 CLAHE 和更高的阈值
        # ),

        # 2. 白天的视频 (用你刚才在第一阶段测出来的新参数)
        (
            "drive8.mp4",
            "Day Highway",
            dict(s_min=120, sx_min=20, use_clahe=True)  # 假设白天需要 CLAHE 和更高的阈值
        ),
    ]

    # 循环处理列表里的每一个视频
    for filename, desc, params in videos:
        print(f"\n=== 正在处理: {desc} ({filename}) ===")

        # 自动拼接路径
        in_path = os.path.join("..", "data", filename)
        # 自动生成不同的输出文件名，避免覆盖
        out_vid = os.path.join("..", "outputs", f"annotated_{filename}")
        out_csv = os.path.join("..", "outputs", f"data_{filename.replace('.mp4', '.csv')}")

        # 检查文件是否存在
        if not os.path.exists(in_path):
            print(f"错误：找不到文件 {in_path}，跳过。")
            continue

        # 启动流水线，传入该视频专属的参数
        run(in_path, out_vid, out_csv,
            s_min=params["s_min"],
            sx_min=params["sx_min"],
            use_clahe=params["use_clahe"],
            nwindows=9, margin=80, minpix=50)  # 这些通常不用变

    print("\n所有视频处理完毕！")

