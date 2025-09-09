#!/usr/bin/env python3
"""
Palm Movement Detection using Optical Flow
AI for Beginners — Computer Vision Lab

This script:
1) Loads video frames (or generates a synthetic demo if no video is supplied).
2) Computes dense optical flow between consecutive frames.
3) Converts flow to polar (magnitude, angle).
4) Builds direction histograms per frame with magnitude thresholding.
5) Plots sample histograms.
6) Infers coarse movement per frame (RIGHT/UP/LEFT/DOWN/STATIC) from histogram energy.

Usage:
  python palm_optical_flow_lab.py --video /path/to/palm.mp4 --outdir ./outputs

If --video is omitted or unreadable, a synthetic "palm" sequence is generated.
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from skimage.registration import optical_flow_tvl1
    HAS_SK_TVL1 = True
except Exception:
    HAS_SK_TVL1 = False


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def to_gray_uint8(frame):
    """Convert an RGB/BGR or single-channel frame to uint8 grayscale."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 3:
        if HAS_CV2:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            r, g, b = arr[...,0], arr[...,1], arr[...,2]
            gray = 0.299*r + 0.587*g + 0.114*b
    elif arr.ndim == 2:
        gray = arr
    else:
        raise ValueError("Unsupported frame shape for grayscale.")
    return ensure_uint8(gray)

def makedirs(p):
    os.makedirs(p, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Get video frames
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_palm_sequence(
    H=240, W=320, rect_size=(70, 100), n_static=5, n_per_dir=12, gap=6
):
    """
    Create synthetic grayscale frames with a filled rectangle "palm" moving:
    RIGHT -> LEFT -> DOWN -> UP, with short static gaps.
    Returns: np.ndarray [T, H, W], dtype=uint8.
    """
    frames = []
    bg = np.full((H, W), 200, dtype=np.uint8)  # light gray background
    ph, pw = rect_size
    y, x = (H - ph) // 2, (W - pw) // 2

    def draw(frame, y0, x0):
        canvas = frame.copy()
        canvas[y0:y0+ph, x0:x0+pw] = 40  # darker palm block
        # add "finger" texture lines
        for k in range(4):
            rr = y0 + 10 + k*12
            canvas[rr:rr+2, x0+10:x0+pw-10] = 60
        return canvas

    def move(dx, dy, n_steps, start_x, start_y):
        locs = []
        cx, cy = start_x, start_y
        for _ in range(n_steps):
            cx += dx
            cy += dy
            cx = max(0, min(W - pw, cx))
            cy = max(0, min(H - ph, cy))
            locs.append((cx, cy))
        return locs

    segments = []
    segments += [(0, 0, n_static)]
    segments += [(+4, 0, n_per_dir)]
    segments += [(0, 0, gap)]
    segments += [(-4, 0, n_per_dir)]
    segments += [(0, 0, gap)]
    segments += [(0, +3, n_per_dir)]
    segments += [(0, 0, gap)]
    segments += [(0, -3, n_per_dir)]
    segments += [(0, 0, gap)]

    cx, cy = x, y
    for dx, dy, n in segments:
        locs = move(dx, dy, n, cx, cy) if (dx != 0 or dy != 0) else [(cx, cy)]*n
        for (cx, cy) in locs:
            frame = draw(bg, cy, cx)
            frames.append(frame)
    return np.stack(frames, axis=0)

def load_video_frames(video_path, max_frames=None, resize_to=None):
    """
    Load frames from a video path using cv2 if available. Returns [T,H,W] grayscale uint8.
    """
    if not HAS_CV2:
        return None
    if not (video_path and os.path.exists(video_path)):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_to is not None:
            frame = cv2.resize(frame, (resize_to[1], resize_to[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Dense optical flow + polar conversion
# ─────────────────────────────────────────────────────────────────────────────

def optical_flow_dense(frames):
    """
    Compute dense optical flow between consecutive frames.
    Returns arrays U, V with shape [T-1, H, W] in pixels/frame:
      - U: horizontal flow (x-axis), positive -> right
      - V: vertical flow (y-axis),   positive -> down
    Priority: cv2 Farnebäck -> scikit-image TV-L1 -> phase correlation fallback.
    """
    T, H, W = frames.shape
    U = np.zeros((T-1, H, W), dtype=np.float32)
    V = np.zeros((T-1, H, W), dtype=np.float32)

    if HAS_CV2:
        fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        for t in range(T-1):
            f0 = frames[t]
            f1 = frames[t+1]
            flow = cv2.calcOpticalFlowFarneback(f0, f1, None, **fb_params)
            U[t] = flow[..., 0]
            V[t] = flow[..., 1]
        return U, V

    if HAS_SK_TVL1:
        for t in range(T-1):
            f0 = frames[t].astype(np.float32) / 255.0
            f1 = frames[t+1].astype(np.float32) / 255.0
            v, u = optical_flow_tvl1(f0, f1)  # returns (V, U)
            U[t] = u
            V[t] = v
        return U, V

    # Fallback: phase correlation (global translation) expanded to dense field
    def phase_corr_shift(im0, im1):
        f0 = np.fft.fft2(im0)
        f1 = np.fft.fft2(im1)
        R = f0 * np.conj(f1)
        eps = np.finfo(np.float64).eps
        R /= np.maximum(np.abs(R), eps)
        r = np.fft.ifft2(R)
        maxima = np.unravel_index(np.argmax(np.abs(r)), r.shape)
        peak = np.array(maxima, dtype=np.float32)
        shifts = np.array(peak, dtype=np.float32)
        shifts[0] = shifts[0] if shifts[0] < im0.shape[0] / 2 else shifts[0] - im0.shape[0]
        shifts[1] = shifts[1] if shifts[1] < im0.shape[1] / 2 else shifts[1] - im0.shape[1]
        dy, dx = shifts  # row, col
        return float(dx), float(dy)

    for t in range(T-1):
        f0 = frames[t].astype(np.float32)
        f1 = frames[t+1].astype(np.float32)
        u, v = phase_corr_shift(f0, f1)
        U[t] = u
        V[t] = v
    return U, V

def flow_to_polar(U, V):
    mag = np.sqrt(U**2 + V**2).astype(np.float32)
    ang = (np.rad2deg(np.arctan2(V, U)) + 360.0) % 360.0  # degrees [0,360)
    return mag, ang


# ─────────────────────────────────────────────────────────────────────────────
# 3) Direction histograms per frame
# ─────────────────────────────────────────────────────────────────────────────

def direction_histogram(ang_deg, mag, mag_thresh=1.0, n_bins=36):
    """
    Weighted histogram over angle with weights = magnitudes.
    Returns (hist, bin_edges).
    """
    mask = mag >= mag_thresh
    if mask.sum() == 0:
        hist = np.zeros(n_bins, dtype=np.float32)
        bin_edges = np.linspace(0, 360, n_bins+1)
        return hist, bin_edges
    angles = ang_deg[mask].ravel()
    weights = mag[mask].ravel()
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 360), weights=weights)
    return hist.astype(np.float32), bin_edges


# ─────────────────────────────────────────────────────────────────────────────
# 4) Plot histograms for sample frames
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(hists, bin_edges, outdir, sample_idxs):
    paths = []
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    for idx in sample_idxs:
        fig = plt.figure(figsize=(7, 3.5))
        plt.bar(centers, hists[idx], width=360/len(hists[idx]))
        plt.xlabel("Direction (degrees)")
        plt.ylabel("Weighted count (by magnitude)")
        plt.title(f"Direction Histogram — Frame pair {idx}→{idx+1}")
        out_path = os.path.join(outdir, f"hist_{idx:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 5) Decide Up/Down/Left/Right from histograms
# ─────────────────────────────────────────────────────────────────────────────

def bins_in_window(center_deg, half_width_deg, n_bins):
    bin_width = 360.0 / n_bins
    centers = (np.arange(n_bins) + 0.5) * bin_width
    low = (center_deg - half_width_deg) % 360.0
    high = (center_deg + half_width_deg) % 360.0
    def in_arc(c):
        if low <= high:
            return (c >= low) & (c <= high)
        else:
            return (c >= low) | (c <= high)
    return np.where(in_arc(centers))[0].tolist()

def infer_directions(hists, n_bins, decision_frac_thresh=0.35):
    half_win = 22.5  # +/- 22.5° windows
    dirs = {
        "RIGHT": bins_in_window(0.0,   half_win, n_bins),
        "UP":    bins_in_window(90.0,  half_win, n_bins),
        "LEFT":  bins_in_window(180.0, half_win, n_bins),
        "DOWN":  bins_in_window(270.0, half_win, n_bins),
    }
    order = ["RIGHT", "UP", "LEFT", "DOWN"]
    scores = np.zeros((hists.shape[0], 4), dtype=np.float32)
    for t in range(hists.shape[0]):
        for j, d in enumerate(order):
            scores[t, j] = hists[t, dirs[d]].sum()
    eps = 1e-6
    tot = (hists.sum(axis=1, keepdims=True) + eps)
    frac = scores / tot
    dec_idx = np.argmax(frac, axis=1)
    dec_val = np.max(frac, axis=1)
    labels = np.array(order, dtype=object)[dec_idx]
    labels[dec_val < decision_frac_thresh] = "STATIC"
    return labels, frac, scores

def summarize_segments(labels):
    segments = []
    if len(labels) == 0:
        return segments
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segments.append((start, i-1, str(cur)))
            cur = labels[i]
            start = i
    segments.append((start, len(labels)-1, str(cur)))
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Palm Movement Detection using Dense Optical Flow")
    parser.add_argument("--video", type=str, default=None, help="Path to input video (optional).")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Directory to save outputs.")
    parser.add_argument("--resize", type=str, default="240x320", help="Resize HxW for processing (e.g., 240x320).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames to read.")
    parser.add_argument("--mag-thresh", type=float, default=0.8, help="Magnitude threshold to ignore tiny motions.")
    parser.add_argument("--bins", type=int, default=36, help="Number of direction bins (e.g., 36 => 10° bins).")
    parser.add_argument("--sample-plots", type=int, nargs="*", default=None, help="Which frame-pair indices to plot.")
    args = parser.parse_args()

    H, W = map(int, args.resize.lower().split("x"))
    makedirs(args.outdir)

    # Load frames
    frames = load_video_frames(args.video, max_frames=args.max_frames, resize_to=(H, W))
    if frames is None:
        print("[Info] No valid video supplied or OpenCV unavailable; generating synthetic demo.")
        frames = generate_synthetic_palm_sequence(H=H, W=W)
    T, H, W = frames.shape
    print(f"[Info] Loaded {T} frames of size {H}x{W} (grayscale).")

    # Optical flow
    U, V = optical_flow_dense(frames)
    mag, ang = flow_to_polar(U, V)
    print(f"[Info] Computed optical flow for {U.shape[0]} frame pairs.")

    # Histograms
    hists = []
    bin_edges = None
    for t in range(mag.shape[0]):
        h, edges = direction_histogram(ang[t], mag[t], mag_thresh=args.mag_thresh, n_bins=args.bins)
        hists.append(h)
        if bin_edges is None:
            bin_edges = edges
    hists = np.stack(hists, axis=0)

    # Plots
    if args.sample_plots is None:
        # pick 4 evenly spaced indices
        sample_idxs = np.linspace(0, hists.shape[0]-1, 4, dtype=int).tolist()
    else:
        sample_idxs = [i for i in args.sample_plots if 0 <= i < hists.shape[0]]
        if not sample_idxs:
            sample_idxs = [0]
    plot_paths = plot_histograms(hists, bin_edges, args.outdir, sample_idxs)
    for p in plot_paths:
        print("[Saved]", p)

    # Infer directions
    labels, frac, scores = infer_directions(hists, args.bins, decision_frac_thresh=0.35)
    segs = summarize_segments(labels)

    # Save CSV
    import csv
    csv_path = os.path.join(args.outdir, "palm_flow_direction_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_pair", "RIGHT_score_frac", "UP_score_frac", "LEFT_score_frac", "DOWN_score_frac", "label"])
        for i in range(len(labels)):
            writer.writerow([i, frac[i,0], frac[i,1], frac[i,2], frac[i,3], labels[i]])
    print("[Saved]", csv_path)

    # Save a brief README
    readme_path = os.path.join(args.outdir, "README_optical_flow_lab.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""Palm Movement Detection using Optical Flow — Lab Outputs

Files:
- palm_flow_direction_scores.csv : Per-frame direction scores (fractions) and predicted labels.
- hist_XXX.png : Direction histograms for sample frame pairs.

Pipeline overview:
1) Load frames (or generate synthetic demo) and convert to grayscale.
2) Compute dense optical flow (Farnebäck if OpenCV is available, else TV-L1 via scikit-image, else global phase correlation fallback).
3) Convert flow (U,V) to polar (magnitude, angle_degrees).
4) Build weighted direction histograms per frame (weights = magnitudes), threshold small vectors (mag_thresh={args.mag_thresh}).
5) Decide coarse movement direction by summing histogram energy around canonical angles {{0°,90°,180°,270°}}.
6) Save and display results.

Command used:
  video={args.video}
  outdir={args.outdir}
  resize={args.resize}
  max_frames={args.max_frames}
  mag_thresh={args.mag_thresh}
  bins={args.bins}
  sample_plots={args.sample_plots}
""")
    print("[Saved]", readme_path)

    # Console summary
    print("\n[Summary] First 40 predictions:")
    for i in range(min(40, len(labels))):
        print(f"  pair {i:03d} → {labels[i]}")

    print("\n[Segments] (start_idx, end_idx, label):")
    for s in segs:
        print(" ", s)


if __name__ == "__main__":
    main()
