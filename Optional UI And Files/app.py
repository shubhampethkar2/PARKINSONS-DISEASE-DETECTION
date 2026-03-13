# ---------------- SYSTEM SETUP ----------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from feature_extractor import extract_features, FEATURE_COLS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroGraph – Parkinson's Detection",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "models/rf_model.pkl"
    if not os.path.exists(model_path):
        st.error(
            "Model file not found at `models/rf_model.pkl`. "
            "Please run `python train_model.py` first."
        )
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 NeuroGraph</h1>
    <h4 style='text-align:center; color:gray;'>
    Parkinson's Disease Detection using Handwriting Analysis
    </h4>
    """,
    unsafe_allow_html=True
)
st.divider()

# ---------------- INPUT & RESULT ----------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 🧍 Upload Handwriting Sample")
    uploaded_file = st.file_uploader(
        "Spiral Drawing Image",
        type=["png", "jpg", "jpeg"]
    )
    predict_btn = st.button("🔍 Analyze", use_container_width=True)

with col2:
    st.markdown("### 🧠 Prediction Outcome")

    if uploaded_file and predict_btn:
        # Decode image
        img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Extract features
        feats = extract_features(img)

        if feats is None:
            st.error("Could not extract features from the image. Please try a clearer spiral drawing.")
        else:
            # Build feature vector in correct column order
            X = np.array([[feats[col] for col in FEATURE_COLS]])

            prob_pd = float(model.predict_proba(X)[0][1])   # probability of Parkinson's

            if prob_pd < 0.33:
                severity = "Healthy / Mild"
                color    = "#2ECC71"
            elif prob_pd < 0.66:
                severity = "Moderate Parkinson's"
                color    = "#F1C40F"
            else:
                severity = "Severe Parkinson's"
                color    = "#E74C3C"

            st.markdown(
                f"<h2 style='color:{color};'>{severity}</h2>",
                unsafe_allow_html=True
            )

            # Confidence bar
            fig, ax = plt.subplots(figsize=(4, 1.5))
            ax.barh([0], [prob_pd * 100], color=color)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Confidence (%)")
            ax.set_title(f"{prob_pd*100:.1f}% Parkinson's Probability")
            st.pyplot(fig)
            plt.close(fig)

            # Show extracted features in expander
            with st.expander("🔬 Extracted Handwriting Features"):
                feat_df = pd.DataFrame([feats]).T.rename(columns={0: "Value"})
                feat_df["Value"] = feat_df["Value"].round(4)
                st.dataframe(feat_df, use_container_width=True)

st.divider()

# ---------------- FEATURE IMPORTANCE (from trained model) ----------------
st.markdown("### 📌 Feature Importance (Model Insight)")

rf_clf = model.named_steps["clf"]
importances = rf_clf.feature_importances_
feat_names  = FEATURE_COLS

# Sort descending
sorted_idx = np.argsort(importances)[::-1]
sorted_imp  = importances[sorted_idx]
sorted_names = [feat_names[i] for i in sorted_idx]

# Donut chart
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.pie(
    sorted_imp,
    labels=sorted_names,
    autopct="%1.0f%%",
    startangle=90,
    wedgeprops=dict(width=0.4)
)
ax2.set_title("Contribution of Handwriting Features")
st.pyplot(fig2)
plt.close(fig2)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "AI-assisted screening tool — Not a medical diagnosis"
    "</p>",
    unsafe_allow_html=True
)
