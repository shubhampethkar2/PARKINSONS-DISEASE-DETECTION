"""
feature_extractor.py
--------------------
Extracts handwriting features from a spiral drawing image.
Features are aligned with the Spiral_HandPD.csv columns.

KEY FIX: All distance/interval stats are computed PER CONTOUR
then aggregated — never across contour boundaries.
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
    gray = cv2.resize(gray, (224, 224))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── RMS of stroke pixel intensities ──────────────────────────────────────
    stroke_pixels = blur[binary > 0].astype(float)
    if stroke_pixels.size == 0:
        return None
    rms = float(np.sqrt(np.mean(stroke_pixels ** 2)))

    # ── Find contours — keep only meaningful ones (area > 5px) ───────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 5 and len(c) > 2]

    if not contours:
        return None

    # ── Per-contour stats ─────────────────────────────────────────────────────
    # For each contour compute inter-point distances (true stroke movement)
    all_distances   = []   # all step distances across all contours
    all_arc_lengths = []   # one arc length per contour
    all_dir_changes = []   # direction change ratio per contour

    for c in contours:
        pts = c.reshape(-1, 2).astype(float)

        # Step distances within this contour only
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        # Remove zero-movement steps (duplicate points from CHAIN_APPROX_NONE)
        dists = dists[dists > 0.5]

        if dists.size > 0:
            all_distances.extend(dists.tolist())

        # Arc length of this stroke
        arc = cv2.arcLength(c, closed=False)
        all_arc_lengths.append(arc)

        # Direction changes in x within this contour
        dx    = diffs[:, 0]
        signs = np.sign(dx)
        signs = signs[signs != 0]
        if signs.size > 1:
            ratio = float(np.sum(signs[1:] != signs[:-1]) / signs.size)
            all_dir_changes.append(ratio)

    if not all_distances:
        return None

    d = np.array(all_distances)
    a = np.array(all_arc_lengths)

    # ── ET-HT proxies (computed properly within contours) ────────────────────
    max_et_ht = float(np.percentile(d, 99))   # use 99th pct to avoid single outliers
    min_et_ht = float(np.percentile(d, 1))    # use 1st pct to avoid zero noise
    std_et_ht = float(d.std())
    mrt       = float(d.mean())

    # ── Hold-time proxies (arc lengths per stroke) ────────────────────────────
    max_ht = float(a.max())
    min_ht = float(a.min())
    std_ht = float(a.std())

    # ── Direction changes (mean ratio across contours) ────────────────────────
    dir_changes = float(np.mean(all_dir_changes)) if all_dir_changes else 0.0

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