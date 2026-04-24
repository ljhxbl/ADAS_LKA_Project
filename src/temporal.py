# src/temporal.py
import numpy as np

class TemporalSmoother:
    def __init__(self, alpha=0.9, th_hi=0.65, th_lo=0.55, keep_last_good=True):
        self.alpha = alpha
        self.th_hi = th_hi
        self.th_lo = th_lo
        self.keep_last_good = keep_last_good
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.last_good_left = None
        self.last_good_right = None
        self.left_state = False
        self.right_state = False

    def _ema(self, prev, cur):
        if prev is None or cur is None:
            return cur
        return self.alpha * np.array(prev) + (1 - self.alpha) * np.array(cur)

    def update(self, lanes):
        # 滞回的 YES/NO
        def hyst(state, conf):
            if conf >= self.th_hi: return True
            if conf <= self.th_lo: return False
            return state

        self.left_state  = hyst(self.left_state,  lanes["left_conf"])
        self.right_state = hyst(self.right_state, lanes["right_conf"])

        # 低置信时用上一次好结果
        left_fit  = lanes["left_fit"]
        right_fit = lanes["right_fit"]

        if self.keep_last_good:
            if lanes["left_conf"]  >= self.th_hi and left_fit is not None:
                self.last_good_left = left_fit
            if lanes["right_conf"] >= self.th_hi and right_fit is not None:
                self.last_good_right = right_fit
            if not self.left_state and self.last_good_left is not None:
                left_fit = self.last_good_left
            if not self.right_state and self.last_good_right is not None:
                right_fit = self.last_good_right

        # EMA 平滑
        left_fit  = self._ema(self.prev_left_fit,  left_fit)
        right_fit = self._ema(self.prev_right_fit, right_fit)
        self.prev_left_fit, self.prev_right_fit = left_fit, right_fit

        # 写回
        lanes["left_fit"]  = left_fit
        lanes["right_fit"] = right_fit
        lanes["left_detected"]  = bool(self.left_state)
        lanes["right_detected"] = bool(self.right_state)
        return lanes
