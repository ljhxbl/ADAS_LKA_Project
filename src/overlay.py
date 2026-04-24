# src/overlay.py
import cv2
import numpy as np
import pandas as pd

class CSVWriter:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def write_row(self, frame_id, left_det, right_det, left_conf, right_conf, lat_offset_m):
        self.rows.append([frame_id,int(left_det),int(right_det),
                          float(left_conf), float(right_conf), float(lat_offset_m)])

    def close(self):
        df = pd.DataFrame(self.rows, columns=[
            "frame_id","left_detected","right_detected","left_conf","right_conf","lat_offset_m"
        ])
        df.to_csv(self.path, index=False)

def _polyline_points(fit, h):
    if fit is None: return None
    ploty = np.linspace(0, h-1, h)
    x = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.vstack((x, ploty)).T.astype(np.float32).reshape(-1,1,2)
    return pts

def _draw_dashed(mask, pts, color, thickness=10, dash=25, gap=20):
    if pts is None: return
    pts = pts.reshape(-1,2).astype(np.int32)
    for i in range(0, len(pts)-1, dash+gap):
        j = min(i+dash, len(pts)-1)
        cv2.polylines(mask, [pts[i:j].reshape(-1,1,2)], False, color, thickness)

def draw_polylines_on_cam(frame_bgr, left_fit, right_fit, Minv,
                          left_on=True, right_on=True):
    h, w = frame_bgr.shape[:2]
    warp_canvas = np.zeros((h,w,3), dtype=np.uint8)

    # 颜色：左=绿，右=蓝，低置信=灰虚线
    left_pts  = _polyline_points(left_fit, h)
    right_pts = _polyline_points(right_fit, h)

    if left_pts is not None:
      if left_on:
        cv2.polylines(warp_canvas, [left_pts.astype(np.int32)], False, (0,255,0), 25)
      else:
        _draw_dashed(warp_canvas, left_pts, (180,180,180), thickness=8)
    if right_pts is not None:
      if right_on:
        cv2.polylines(warp_canvas, [right_pts.astype(np.int32)], False, (255,0,0), 25)
      else:
        _draw_dashed(warp_canvas, right_pts, (180,180,180), thickness=8)

    unwarp = cv2.warpPerspective(warp_canvas, Minv, (w,h))
    out = frame_bgr.copy()
    mask = unwarp>0
    out[mask] = unwarp[mask]
    return out

def draw_hud(out_img, left_det, right_det, conf_avg, takeover=False):
    text = f"Left: {'YES' if left_det else 'NO'} | Right: {'YES' if right_det else 'NO'} | Conf: {conf_avg:.2f}"
    cv2.rectangle(out_img, (10,10), (700,60), (0,0,0), -1)
    cv2.putText(out_img, text, (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    if takeover:
        cv2.rectangle(out_img, (out_img.shape[1]-330,10), (out_img.shape[1]-10,60), (0,0,0), -1)
        cv2.putText(out_img, "Driver Takeover", (out_img.shape[1]-320,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2, cv2.LINE_AA)

def estimate_lateral_offset_m(lfit, rfit, frame_shape,
                              meters_per_pix_x=3.7/700):
    h, w = frame_shape[:2]
    if lfit is None or rfit is None:
        return 0.0
    y = h - 1
    xl = lfit[0]*y*y + lfit[1]*y + lfit[2]
    xr = rfit[0]*y*y + rfit[1]*y + rfit[2]
    lane_center = (xl + xr) / 2.0
    img_center  = w / 2.0
    dx_pix = lane_center - img_center
    return float(dx_pix * meters_per_pix_x)
