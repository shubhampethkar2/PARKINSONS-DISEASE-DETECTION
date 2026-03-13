"""
train_model.py
--------------
Trains a Random Forest classifier on the combined dataset
(original Spiral_HandPD CSV + synthetic PD samples).

Usage
-----
    python train_model.py \
        --csv   Spiral_HandPD_combined.csv \
        --imgs  images/ \
        --out   models/rf_model.pkl

Defaults:
    --csv   Spiral_HandPD_combined.csv   ← updated to use combined dataset
    --imgs  images/
    --out   models/rf_model.pkl
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

from feature_extractor import extract_features, FEATURE_COLS


# ── Label mapping: CSV uses 1=Parkinson's, 2=Healthy → remap to 1=PD, 0=Healthy
LABEL_MAP = {1: 1, 2: 0}


# ────────────────────────────────────────────────────────────────────────────
# 1.  Load CSV (works for original, synthetic, or combined CSV)
# ────────────────────────────────────────────────────────────────────────────
def load_csv_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    missing = [c for c in FEATURE_COLS + ["CLASS_TYPE"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df[FEATURE_COLS + ["CLASS_TYPE", "IMAGE_NAME"]].copy()
    df["label"]  = df["CLASS_TYPE"].map(LABEL_MAP)
    df["source"] = "csv"
    df.drop(columns=["CLASS_TYPE"], inplace=True)
    df.dropna(subset=["label"], inplace=True)

    pd_count  = int(df["label"].sum())
    hlt_count = int((df["label"] == 0).sum())
    print(f"[CSV]  Loaded {len(df)} rows  |  PD={pd_count}  Healthy={hlt_count}")

    # Warn if heavily imbalanced
    ratio = pd_count / max(hlt_count, 1)
    if ratio < 0.3 or ratio > 3.0:
        print(f"[WARN] Class imbalance detected (PD/Healthy ratio={ratio:.2f}). "
              f"Consider adding more synthetic samples.")
    return df


# ────────────────────────────────────────────────────────────────────────────
# 2.  Extract features from matched images (optional augmentation)
# ────────────────────────────────────────────────────────────────────────────
def load_image_data(img_dir: str, csv_image_names: set) -> pd.DataFrame:
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print(f"[IMG]  Folder '{img_dir}' not found — skipping image extraction.")
        return pd.DataFrame()

    rows    = []
    matched = 0
    skipped = 0

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for fpath in img_dir.glob(ext):
            fname = fpath.name
            if fname not in csv_image_names:
                skipped += 1
                continue
            feats = extract_features(str(fpath))
            if feats is None:
                continue
            feats["IMAGE_NAME"] = fname
            feats["label"]      = None   # filled from CSV later
            feats["source"]     = "image"
            rows.append(feats)
            matched += 1

    print(f"[IMG]  Matched {matched} images to CSV  |  {skipped} unlabelled images skipped")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ────────────────────────────────────────────────────────────────────────────
# 3.  Merge CSV + image features
# ────────────────────────────────────────────────────────────────────────────
def build_dataset(csv_df: pd.DataFrame, img_df: pd.DataFrame) -> pd.DataFrame:
    if img_df.empty:
        print("[MERGE] Using CSV data only.")
        return csv_df[FEATURE_COLS + ["label"]].dropna()

    label_map_by_name        = dict(zip(csv_df["IMAGE_NAME"], csv_df["label"]))
    img_df["label"]          = img_df["IMAGE_NAME"].map(label_map_by_name)

    combined = pd.concat(
        [csv_df[FEATURE_COLS + ["label"]], img_df[FEATURE_COLS + ["label"]]],
        ignore_index=True
    )
    combined.dropna(inplace=True)

    pd_count  = int(combined["label"].sum())
    hlt_count = int((combined["label"] == 0).sum())
    print(f"[MERGE] Final dataset: {len(combined)} rows  |  PD={pd_count}  Healthy={hlt_count}")
    return combined


# ────────────────────────────────────────────────────────────────────────────
# 4.  Train & evaluate
# ────────────────────────────────────────────────────────────────────────────
def train(dataset: pd.DataFrame, out_path: str):
    X = dataset[FEATURE_COLS].values
    y = dataset["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 5-fold cross validation
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n[TRAIN] 5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"[TEST]  Accuracy : {acc:.4f}")
    print(f"[TEST]  ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    rf          = pipeline.named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string())

    # Save model + metadata for app to display
    model_meta = {
        "pipeline":         pipeline,
        "train_accuracy":   acc,
        "train_auc":        auc,
        "cv_auc_mean":      float(cv_scores.mean()),
        "cv_auc_std":       float(cv_scores.std()),
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "n_pd":             int(y.sum()),
        "n_healthy":        int((y == 0).sum()),
        "feature_cols":     FEATURE_COLS,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(model_meta, out_path)
    print(f"\n✅  Model + metadata saved → {out_path}")
    return pipeline


# ────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train NeuroGraph RF model")
    parser.add_argument("--csv",  default="Spiral_HandPD_combined.csv",
                        help="Path to combined CSV (original + synthetic PD)")
    parser.add_argument("--imgs", default="images",
                        help="Folder with spiral images (optional)")
    parser.add_argument("--out",  default="models/rf_model.pkl",
                        help="Output model path")
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroGraph — Training Pipeline v2")
    print("=" * 60)
    print(f"  CSV  : {args.csv}")
    print(f"  IMGS : {args.imgs}")
    print(f"  OUT  : {args.out}")
    print("=" * 60)

    csv_df    = load_csv_data(args.csv)
    csv_names = set(csv_df["IMAGE_NAME"].tolist())
    img_df    = load_image_data(args.imgs, csv_names)
    dataset   = build_dataset(csv_df, img_df)

    train(dataset, args.out)


if __name__ == "__main__":
    main()
