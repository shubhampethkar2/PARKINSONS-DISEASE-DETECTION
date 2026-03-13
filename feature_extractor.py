"""
feature_extractor.py  v3
------------------------
Extracts 9 handwriting features from a spiral drawing image.
All values are in IMAGE-SPACE units (pixels) — consistent between
training (from image files) and inference (from uploaded images).

IMPORTANT: This extractor must be used for BOTH training and inference.
Do NOT mix with CSV-based feature values — they are in tablet-sensor units
which are completely different scales.
"""

import cv2
import numpy as np

FEATURE_COLS = [
    "RMS",
    "MAX_BETWEEN_ET_HT",
    "MIN_BETWEEN_ET_HT",
    "STD_DEVIATION_ET_HT",
    "MRT",
    "MAX_HT",
    "MIN_HT",
    "STD_HT",
    "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT",
]


def extract_features(img_input):
    # ── Load ─────────────────────────────────────────────────────────────────
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            return None
    else:
        img = img_input.copy()

    # ── Pre-process ──────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Normalise to 256x256 consistently
    gray = cv2.resize(gray, (256, 256))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── RMS of stroke pixel intensities (pen pressure proxy) ─────────────────
    stroke_pixels = blur[binary > 0].astype(float)
    if stroke_pixels.size == 0:
        return None
    rms = float(np.sqrt(np.mean(stroke_pixels ** 2)))

    # ── Find contours — filter noise (area > 10px, length > 3 pts) ───────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 10 and len(c) > 3]

    if not contours:
        return None

    # ── Per-contour stats ────────────────────────────────────────────────────
    all_step_dists  = []   # step-by-step distances within each contour
    all_arc_lengths = []   # total arc length per contour
    all_dir_ratios  = []   # direction-change ratio per contour

    for c in contours:
        pts   = c.reshape(-1, 2).astype(float)
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        # Remove duplicate/zero-movement points
        dists = dists[dists > 0.3]
        if dists.size > 2:
            all_step_dists.extend(dists.tolist())

        # Arc length of this stroke
        arc = cv2.arcLength(c, closed=False)
        if arc > 1.0:
            all_arc_lengths.append(arc)

        # Direction changes in x within this contour
        dx    = diffs[:, 0]
        signs = np.sign(dx)
        signs = signs[signs != 0]
        if signs.size > 2:
            ratio = float(np.sum(signs[1:] != signs[:-1]) / signs.size)
            all_dir_ratios.append(ratio)

    if not all_step_dists or not all_arc_lengths:
        return None

    d = np.array(all_step_dists)
    a = np.array(all_arc_lengths)

    # ── Feature computation ──────────────────────────────────────────────────
    # ET-HT proxies: statistics of step distances within strokes
    max_et_ht = float(np.percentile(d, 95))   # 95th pct — robust max
    min_et_ht = float(np.percentile(d, 5))    # 5th pct  — robust min
    std_et_ht = float(d.std())                 # variability of movement
    mrt       = float(d.mean())               # mean step size

    # Hold-time proxies: statistics of stroke arc lengths
    max_ht = float(a.max())
    min_ht = float(a.min())
    std_ht = float(a.std())

    # Direction changes
    dir_changes = float(np.mean(all_dir_ratios)) if all_dir_ratios else 0.0

    return {
        "RMS":                                               rms,
        "MAX_BETWEEN_ET_HT":                                max_et_ht,
        "MIN_BETWEEN_ET_HT":                                min_et_ht,
        "STD_DEVIATION_ET_HT":                              std_et_ht,
        "MRT":                                              mrt,
        "MAX_HT":                                           max_ht,
        "MIN_HT":                                           min_ht,
        "STD_HT":                                           std_ht,
        "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT":  dir_changes,
    }
