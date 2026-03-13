"""
train_model.py
--------------
Combines the Spiral_HandPD CSV with features extracted from local images,
then trains a Random Forest classifier and saves it as models/rf_model.pkl.

Usage
-----
    python train_model.py \
        --csv   data/Spiral_HandPD.csv \
        --imgs  data/images \
        --out   models/rf_model.pkl

Defaults:
    --csv   Spiral_HandPD.csv
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

from feature_extractor import extract_features, FEATURE_COLS


# ── Label mapping: CSV uses 1=Parkinson, 2=Healthy → remap to 1=PD, 0=Healthy
LABEL_MAP = {1: 1, 2: 0}


# ────────────────────────────────────────────────────────────────────────────
# 1.  Load CSV rows
# ────────────────────────────────────────────────────────────────────────────
def load_csv_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[FEATURE_COLS + ["CLASS_TYPE", "IMAGE_NAME"]].copy()
    df["label"] = df["CLASS_TYPE"].map(LABEL_MAP)
    df["source"] = "csv"
    df.drop(columns=["CLASS_TYPE"], inplace=True)
    print(f"[CSV]  Loaded {len(df)} rows  |  PD={df['label'].sum()}  Healthy={(df['label']==0).sum()}")
    return df


# ────────────────────────────────────────────────────────────────────────────
# 2.  Extract features from images NOT already in the CSV
# ────────────────────────────────────────────────────────────────────────────
def load_image_data(img_dir: str, csv_image_names: set) -> pd.DataFrame:
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print(f"[IMG]  Folder '{img_dir}' not found — skipping image extraction.")
        return pd.DataFrame()

    rows = []
    skipped = 0

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for fpath in img_dir.glob(ext):
            fname = fpath.name

            # Skip images already covered by the CSV
            if fname in csv_image_names:
                continue

            # We cannot determine the label for unknown images
            # → skip them (no label = can't train)
            skipped += 1

    # Also process images that ARE in the CSV directory
    # (acts as a secondary feature source / consistency check)
    matched = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for fpath in img_dir.glob(ext):
            fname = fpath.name
            if fname not in csv_image_names:
                continue
            feats = extract_features(str(fpath))
            if feats is None:
                continue
            feats["IMAGE_NAME"] = fname
            feats["label"] = None          # will be merged from CSV
            feats["source"] = "image"
            rows.append(feats)
            matched += 1

    print(f"[IMG]  Matched {matched} images to CSV  |  {skipped} unlabelled images skipped")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ────────────────────────────────────────────────────────────────────────────
# 3.  Merge & deduplicate
# ────────────────────────────────────────────────────────────────────────────
def build_dataset(csv_df: pd.DataFrame, img_df: pd.DataFrame) -> pd.DataFrame:
    if img_df.empty:
        print("[MERGE] Using CSV data only.")
        return csv_df[FEATURE_COLS + ["label"]].dropna()

    # Fill image labels from CSV
    label_map_by_name = dict(zip(csv_df["IMAGE_NAME"], csv_df["label"]))
    img_df["label"] = img_df["IMAGE_NAME"].map(label_map_by_name)

    # Concatenate; keep CSV rows as authoritative, image rows as augmentation
    combined = pd.concat(
        [csv_df[FEATURE_COLS + ["label"]], img_df[FEATURE_COLS + ["label"]]],
        ignore_index=True
    )
    combined.dropna(inplace=True)
    print(f"[MERGE] Final dataset: {len(combined)} rows  |  PD={combined['label'].sum()}  Healthy={(combined['label']==0).sum()}")
    return combined


# ────────────────────────────────────────────────────────────────────────────
# 4.  Train
# ────────────────────────────────────────────────────────────────────────────
def train(dataset: pd.DataFrame, out_path: str):
    X = dataset[FEATURE_COLS].values
    y = dataset["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n[TRAIN] 5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    # Evaluation on held-out test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n[TEST]  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"[TEST]  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # Feature importances
    rf = pipeline.named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string())

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"\n✅  Model saved → {out_path}")

    return pipeline, importances


# ────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Parkinson's RF model")
    parser.add_argument("--csv",  default="Spiral_HandPD.csv",  help="Path to CSV file")
    parser.add_argument("--imgs", default="images",             help="Folder with spiral images")
    parser.add_argument("--out",  default="models/rf_model.pkl",help="Output model path")
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroGraph — Training Pipeline")
    print("=" * 60)

    csv_df = load_csv_data(args.csv)
    csv_names = set(csv_df["IMAGE_NAME"].tolist())

    img_df = load_image_data(args.imgs, csv_names)
    dataset = build_dataset(csv_df, img_df)

    train(dataset, args.out)


if __name__ == "__main__":
    main()
