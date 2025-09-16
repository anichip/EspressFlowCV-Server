import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random

# ---------------------------
# Configuration dataclasses
# ---------------------------

# WHERE do I look for the espresso streams? --> Center rectangular area

@dataclass
class ROIConfig:
    # Fractional ROI (x0, y0, x1, y1) relative to frame size
    # (top-left roi 0 , bottom-right roi 1)
    x0: float = 0.11
    y0: float = 0.17
    x1: float = 0.86
    y1: float = 0.55

# What are the qualifications for being espresso? 

@dataclass
class Thresholds:
    # Motion
    flow_mag_thresh: float = 1.3 #min optical flow magnitude to consider pixels moving
    min_area: int = 205   # in pixels, after ROI. Throw out those tiny specks. Have to be large enough to be considered
    min_height: int = 35  # streams are tall
    min_aspect: float = 1.5  # h/w ratio (higher means tall & thin)

    # Color (HSV in OpenCV ranges: H:0..180, S:0..255, V:0..255)
    h_lo: int = 8     # "light brown/amber" lower hue 
    h_hi: int = 28    # "rich brown/amber" upper hue
    #these bottom three are for excluding noise
    s_lo: int = 75   
    v_lo: int = 25  
    v_hi: int = 190

    position_bias: str = "offcenter"  # options: "center" | "neutral" | "offcenter"

    #<>TROUBLESHOOTING<>#
    #too many false positives from metal glare? --> raise s_lo and lower v_hi (less bright stuff allowed)
    #no stream detected at all? --> lower flow_mag_thresh and widen the HSV window
    #Increase min_aspect and min_area if stream is thicker than anticipated

    # Post timelines
    onset_area_px: int = 300  # first time the detected component reaches this area, we say 'flow starts'

    #Overall, "are you moving?" --> "do you look like espresso color?" --> "are you thin?" --> ignore everything else

# For giving us kymograph and debug overlay video. 

@dataclass
class DebugConfig:
    save_overlay_video: bool = True
    overlay_fps: int = 30 #playback fps in visual
    save_kymograph: bool = True 

# For each stream component per frame, what did we detect and where?

@dataclass
class FrameStats:
    stream_found: bool
    area: int
    width: int
    height: int
    cx: float
    cy: float
    hue_med: float
    val_med: float


# ---------------------------
# Utilities
# ---------------------------

# Covert ROI fractions into pixel coordinates for open CV
# Input: image shape (from np array most likely), ROIConfig object with fractions 
# Output: Coordinates 

def _roi_rect(shape, roi_cfg: ROIConfig) -> Tuple[int, int, int, int]:
    H, W = shape[:2] 
    x0 = int(W * roi_cfg.x0) 
    y0 = int(H * roi_cfg.y0)
    x1 = int(W * roi_cfg.x1)
    y1 = int(H * roi_cfg.y1)
    return x0, y0, x1, y1 #casted to int so it makes a pixel value

# Score a detected blob stream based on how big and how close to center
# Input: horizontal center, roi width, area, 
# Output: Component score

def _component_score(cx, roi_w, area, aspect, bias="neutral"):

    # base: larger area and taller/thinner shapes are better
    base = area * max(1e-6, aspect)

    center = roi_w / 2.0
    dist = abs(cx - center) / max(1.0, roi_w / 2.0)
    edge_penalty = 0.7 #our videos are centered, so 70% is okay for being that much off center

    if bias == "center" :
        return base * (1.0 - edge_penalty * dist)
    elif bias == "offcenter":
        offcenter_bonus = 0.7 + 0.3 * dist               # 0.7..1.0
        return base * offcenter_bonus
    else:
        #neutral
        return base

# Given all candidate contours from "motion and color" mask, pick the blob that looks most like a fallen stream
# Input: Roi width, contours from cv2, Thresholds object with the qualifications for an espresso stream
# Output: 

def _best_component(contours, roi_w, thr: Thresholds):

    bias = thr.position_bias

    best = None
    best_score = 0.0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < thr.min_height or w < 2:
            continue
        area = int(cv2.contourArea(c))
        if area < thr.min_area:
            continue
        aspect = h / max(1.0, w)
        if aspect < thr.min_aspect:
            continue
        cx = x + w / 2.0
        score = _component_score(cx, roi_w, area, aspect, bias) #^
        if score > best_score:
            best_score = score
            best = (x, y, w, h, area, cx, y + h/2.0)
    return best

# It's a super compact visualization of stream stability on every frame. PIXEL Intensity

# Kind of like a tree's rings but faster. Imagine you freeze time every 1/60th second 
# and just record “how dark is each column under the spouts?” 
# Then you stack those snapshots top-to-bottom. 
# You’ve turned a video into a barcode-like picture that shows the flow’s story at a glance.

def _make_kymograph(columns_over_time: List[np.ndarray]) -> np.ndarray:
    if not columns_over_time:
        return np.zeros((10, 10), dtype=np.uint8) 
    M = np.stack(columns_over_time, axis=0)
    M = M.astype(np.float32)
    # normalize each row 0..255 for visibility
    row_min = M.min(axis=1, keepdims=True)
    row_max = M.max(axis=1, keepdims=True)
    Mn = (M - row_min) / np.maximum(1e-6, (row_max - row_min))
    return (Mn * 255).astype(np.uint8)

# one thing to note: if there are big background bands in the kymograph,...
# narrow ROIConfig horizontally (e.g., x0=0.42, x1=0.58).
# Increase Thresholds.s_lo (e.g., 60–80) and lower v_hi (≤200) to cut bright metal.

# to not mess up consistency if two streams are detected and it occasionally switches

def medfilt1_nan(x, k=5):
    """
    Median filter for 1D arrays that gracefully handles NaNs.
    - Ignores NaNs inside each window (uses np.nanmedian)
    - If a window is all-NaN, falls back to nearest non-NaN via edge padding
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1  # force odd
    pad = k // 2

    # Edge-pad with nearest real values to avoid all-NaN windows at the ends
    # (we’ll still respect NaNs inside the valid region)
    # Make a copy we can pad from (using nearest non-NaN neighbors)
    x_filled = x.copy()
    # If the entire array is NaN, just return zeros
    if np.all(np.isnan(x_filled)):
        return np.zeros_like(x_filled)

    # Forward-fill then back-fill to get a nearest-neighbor non-NaN series
    # (without changing the original x; this is only for padding)
    ff = x_filled.copy()
    last = np.nan
    for i in range(len(ff)):
        if not np.isnan(ff[i]):
            last = ff[i]
        elif not np.isnan(last):
            ff[i] = last
    bf = ff.copy()
    last = np.nan
    for i in range(len(bf)-1, -1, -1):
        if not np.isnan(bf[i]):
            last = bf[i]
        elif not np.isnan(last):
            bf[i] = last
    nn = bf  # nearest-neighbor filled version for edges

    # Build the padded array using nearest non-NaN values for the edges
    xp = np.pad(nn, (pad, pad), mode="edge")

    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        w = xp[i:i+k].copy()

        # Inside the window, we want the real data when available.
        # Replace the padded positions with the original x where present.
        # (This preserves NaNs inside the valid region for nanmedian to ignore.)
        center = i + pad
        # The valid region in xp that corresponds to real indices:
        # indices [pad, pad+len(x)-1] map to x[0..len(x)-1]
        lo = pad
        hi = pad + len(x) - 1
        idxs = np.arange(i, i+k)
        real = (idxs >= lo) & (idxs <= hi)
        w[real] = x[idxs[real] - pad]

        med = np.nanmedian(w)
        if np.isnan(med):
            # If everything in the window is NaN, fall back to nearest non-NaN in nn
            med = nn[i]
        out[i] = med
    return out


# ---------------------------
# Core per-frame segmentation
# ---------------------------

# This is the HEART! 
# Per-video segmenter that remembers the previous grayscale ROI so that we can compute optical flow betweeen consecutive frames

class EspressoStreamSegmenter:
    def __init__(self, roi_cfg: ROIConfig, thr: Thresholds):
        self.roi_cfg = roi_cfg
        self.thr = thr
        self.prev_gray_roi = None # the previous gray roi is none by default for the first frame. Then it changes

    # The segmenting function:
    # Given one frame, crop ROI --> compute motion + masks --> combine --> pick best blob --> get per-frame statistics

    def segment(self, frame_bgr: np.ndarray) -> Tuple[FrameStats, np.ndarray, Tuple[int,int,int,int]]:
        H, W = frame_bgr.shape[:2]
        x0, y0, x1, y1 = _roi_rect(frame_bgr.shape, self.roi_cfg)
        roi = frame_bgr[y0:y1, x0:x1] #OFFICIAL
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        #optical flow --> motion magnitude
        if self.prev_gray_roi is None or self.prev_gray_roi.shape != gray.shape:
            #no previous frame or mismatch , so fill the magnitude with 0s
            flow_mag = np.zeros_like(gray, dtype=np.float32)
        else:
            #estimate motion between two grayscale frames
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray_roi, gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx, fy = flow[..., 0], flow[..., 1]
            flow_mag = np.sqrt(fx * fx + fy * fy)

        self.prev_gray_roi = gray 

        #remember the flow_mag_thresh for what counts as moving. If it is greater than that, turn to white. everything else black
        motion_mask = (flow_mag > self.thr.flow_mag_thresh).astype(np.uint8) * 255
        # Ensures the moving liquid is treated as one vertical blob, not a stack of disconnected patches.
        motion_mask = cv2.dilate(motion_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)), 1)

        color_mask = cv2.inRange(hsv,
                                 (self.thr.h_lo, self.thr.s_lo, self.thr.v_lo),
                                 (self.thr.h_hi, 255, self.thr.v_hi))

        #<>CAN COMMENT OUT IF IT DOESN'T WORK
        # Remove very bright, low-saturation pixels (chrome reflections)
        glare_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
        color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(glare_mask))

        stream_mask = cv2.bitwise_and(motion_mask, color_mask)

        # stream_mask = cv2.dilate(stream_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)

        contours, _ = cv2.findContours(stream_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #this gives us input for _best_component
        best = _best_component(contours, roi_w, self.thr) #best part of the stream contour

        if best is None:
            return FrameStats(False, 0, 0, 0, -1, -1, np.nan, np.nan), stream_mask, (x0, y0, x1, y1)

        x, y, w, h, area, cx, cy = best #the winning blob's bounding box + good info

        comp_mask = np.zeros_like(stream_mask)
        cv2.rectangle(comp_mask, (x, y), (x + w, y + h), 255, -1)
        comp_mask = cv2.bitwise_and(comp_mask, stream_mask)

        comp_inds = comp_mask > 0
        hue_med = float(np.median(hsv[...,0][comp_inds])) if np.any(comp_inds) else np.nan
        val_med = float(np.median(hsv[...,2][comp_inds])) if np.any(comp_inds) else np.nan

        #And now, we got descriptive info for one frame. return it and move on to the next one
        return FrameStats(True, int(area), int(w), int(h), float(cx), float(cy), hue_med, val_med), stream_mask, (x0, y0, x1, y1)


# ---------------------------
# Timeline feature extraction
# ---------------------------

# find frame where the stream area is big enough to say "oh okay flow has started"

def _first_onset(area_t: List[int], px_thresh: int) -> Optional[int]:
    for i, a in enumerate(area_t):
        if a >= px_thresh:
            return i
    return None

# linear trend of sequence with least squares. "Is the stream getting narrower or wider as we increment time?"

def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float32)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y.astype(np.float32), rcond=None)[0]
    return float(m)

# Coefficient of Variation (std/mean). "How jittery is the width compared to the size?"

def _cv(x: np.ndarray) -> float:
    mu = np.mean(x) if len(x) else 0.0
    sd = np.std(x) if len(x) else 0.0
    return float(sd / (mu + 1e-6))

#this is the main thing we run at the end. Timelines turn into features for ML

def extract_features_from_timelines(width_t: List[int],
                                    area_t: List[int],
                                    cx_t: List[float],
                                    hue_t: List[float],
                                    val_t: List[float],
                                    fps: int) -> Dict[str, float]:
    
    W = np.array(width_t, dtype=np.float32)
    A = np.array(area_t, dtype=np.float32)
    CX = np.array(cx_t, dtype=np.float32)
    H = np.array(hue_t, dtype=np.float32)
    V = np.array(val_t, dtype=np.float32)

    #detect first real flow
    onset = _first_onset(area_t, px_thresh=Thresholds().onset_area_px)
    onset_time = (onset / fps) if onset is not None else np.nan

    if onset is not None:
        #save that onset onward
        W2, A2, CX2, H2, V2 = W[onset:], A[onset:], CX[onset:], H[onset:], V[onset:]
    else:
        W2, A2, CX2, H2, V2 = W, A, CX, H, V

    # Feature: Continuity
    cont = float(np.mean(A2 > Thresholds().onset_area_px)) if len(A2) else 0.0

    # Features: Width and stability trend, so amplitude, average width, side-side wobble (jitter_cx)
    mean_w = float(np.mean(W2)) if len(W2) else 0.0
    cv_w = _cv(W2) if len(W2) else 0.0
    amp_w = float(np.max(W2) - np.min(W2)) if len(W2) else 0.0
    slope_w = _slope(W2) if len(W2) else 0.0
    jitter_cx = float(np.std(CX2)) if len(CX2) else 0.0

    def thirds_delta(arr):
        if len(arr) < 9:
            return np.nan
        n = len(arr)
        a = np.nanmedian(arr[: n//3])
        b = np.nanmedian(arr[- n//3:])
        return float(b - a)

    val_delta = thirds_delta(V2) # Feature: Change in brigthness 
    hue_delta = thirds_delta(H2) # Feature: Change in Hue

    if len(A2) > 3:
        onoff = (A2 > Thresholds().onset_area_px).astype(np.int32)
        flicker = int(np.sum(np.abs(np.diff(onoff)) == 1))
    else:
        flicker = 0

    #deliver the package
    return {
        "onset_time_s": onset_time,
        "continuity": cont,
        "mean_width": mean_w,
        "cv_width": cv_w,
        "amp_width": amp_w,
        "slope_width": slope_w,
        "jitter_cx": jitter_cx,
        "delta_val": val_delta,
        "delta_hue": hue_delta,
        "flicker": float(flicker),
    }

#For the computer. But this is probably going to be dropped during Ml. might comment this out.
def simple_rule_classifier(features: Dict[str, float]) -> str:
    cont = features["continuity"]
    mean_w = features["mean_width"]
    cv_w = features["cv_width"]
    val_delta = features["delta_val"]
    slope_w = features["slope_width"]

    if (cont < 0.55 and mean_w < 6) or (val_delta is not None and not np.isnan(val_delta) and val_delta > -5):
        return "underextracted"
    if (val_delta is not None and not np.isnan(val_delta) and val_delta < -25) and slope_w < -0.02:
        return "overextracted"
    if cont >= 0.7 and cv_w < 0.35 and mean_w >= 6:
        return "perfect_or_mid"

    return "mid"

# ---------------------------
# Main pipeline for a folder of frames
# ---------------------------

# Input: path, fps, duration, ROIConfig object, Thresholds Object, DebugConfig object
# Output: feature arrays for the csv

def process_frames_folder(folder: str,
                          fps: int = 60,
                          max_seconds: float = 7.0,
                          roi_cfg: ROIConfig = ROIConfig(),
                          thr: Thresholds = Thresholds(),
                          debug: DebugConfig = DebugConfig()) -> Dict[str, float]:
    
    frame_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg") and f.lower() != "_kymograph.png"])

    if not frame_files:
        raise FileNotFoundError(f"No .jpg frames found in {folder}")

    max_frames = int(fps * max_seconds)
    frame_files = frame_files[:max_frames]

    seg = EspressoStreamSegmenter(roi_cfg, thr) 

    width_t, area_t, cx_t, hue_t, val_t = [], [], [], [], []
    columns_for_kymo: List[np.ndarray] = []
    writer = None  # Initialize VideoWriter for debug overlay

    for idx, fname in enumerate(frame_files):
        path = os.path.join(folder, fname)
        frame = cv2.imread(path) #open the photo
        if frame is None:
            continue

        stats, stream_mask, roi_rect = seg.segment(frame)
        x0, y0, x1, y1 = roi_rect

        #accumulate time series
        width_t.append(stats.width)
        area_t.append(stats.area)
        cx_t.append(stats.cx)
        hue_t.append(stats.hue_med)
        val_t.append(stats.val_med)

        #collect columns for kymograph (darker --> larger)
        if debug.save_kymograph:
            roi = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # gray = cv2.medianBlur(gray,3) #noise reduction #<>IF NEEDED<>#
            columns_for_kymo.append(np.sum(255 - gray, axis=0))

        #as far as overlays, I don't want everyone to have an overlay. I just want some of them to have an overlay. 
        # rand = random.random()
        # debug.save_overlay_video = rand < 0.25 #about 25% probability

        #new approach: I always wanted to choose which ones I wanted to debug. Just these 6 for now
        select_vids = ["frames_good_pulls/vid_2_good","frames_good_pulls/vid_14_good","frames_good_pulls/vid_47_good","frames_good_pulls/vid_97_good","frames_under_pulls/vid_18_under","frames_under_pulls/vid_74_under"]
        debug.save_overlay_video = folder in select_vids

        #write the debug overlay video to evaluate
        if debug.save_overlay_video:
            overlay = frame.copy()
            # draw ROI
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # draw detected stream bbox (project ROI coords back to full image)
            if stats.stream_found and stats.width > 0 and stats.height > 0:
                rx = int(x0 + stats.cx - stats.width / 2)
                ry = int(y0 + stats.cy - stats.height / 2)
                cv2.rectangle(overlay, (rx, ry), (rx + stats.width, ry + stats.height), (255, 0, 0), 2)
            # HUD text
            vtxt = "nan" if np.isnan(stats.val_med) else f"{stats.val_med:.1f}"
            txt = f"W:{stats.width} A:{stats.area} Vmed:{vtxt}"
            cv2.putText(overlay, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = os.path.join(folder, "_debug_overlay.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, debug.overlay_fps, (overlay.shape[1], overlay.shape[0]))
            writer.write(overlay)

    if writer is not None:
        writer.release()

    # save kymograph image
    if debug.save_kymograph and len(columns_for_kymo) > 5:
        kymo = _make_kymograph(columns_for_kymo)
        kymo_path = os.path.join(folder, "_kymograph.png")
        cv2.imwrite(kymo_path, kymo)

    # After collecting timelines...
    width_t_smooth = medfilt1_nan(width_t, k=5)
    cx_t_smooth = medfilt1_nan(cx_t, k=5)
    hue_t_smooth = medfilt1_nan(hue_t, k=5)
    val_t_smooth = medfilt1_nan(val_t, k=5)

    feats = extract_features_from_timelines(width_t_smooth, area_t, cx_t_smooth, hue_t_smooth, val_t_smooth, fps=fps)
    feats["label_rule_based"] = simple_rule_classifier(feats)
    return feats
