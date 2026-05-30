import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ---------- Page configuration ----------
st.set_page_config(page_title="Diagnostic Assistant", page_icon="🔬", layout="centered")

# ---------- Load and prepare data ----------
@st.cache_data
def load_data():
    # Adjust this path to where your data.csv is located
    df = pd.read_csv("Data/data.csv")
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, list(X.columns)

# ---------- Prediction function ----------
def predict(features, model, scaler, feature_names):
    input_df = pd.DataFrame([features], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][pred]
    return pred, proba

# ---------- Main UI ----------
def main():
    # Title and description
    st.title("🔬 Medical Diagnostic Support Tool")
    st.markdown("Enter patient measurements below to receive a diagnostic prediction.")
    
    # Load model and data
    model, scaler, feature_names = train_model()
    data = load_data()
    
    # Get min, max, default for sliders from training data
    min_vals = data[feature_names].min()
    max_vals = data[feature_names].max()
    default_vals = data[feature_names].median()
    
    # Prediction area (shown first, no scroll)
    st.subheader("📊 Prediction Result")
    result_placeholder = st.empty()
    confidence_placeholder = st.empty()
    
    # Sliders inside an expander (collapsed by default to save space)
    with st.expander("✏️ Adjust clinical measurements", expanded=True):
        cols = st.columns(3)
        user_input = {}
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                user_input[feature] = st.slider(
                    label=feature.replace("_", " ").title(),
                    min_value=float(min_vals[feature]),
                    max_value=float(max_vals[feature]),
                    value=float(default_vals[feature]),
                    key=f"slider_{feature}",
                    help="Drag to adjust value"
                )
    
    # Predict button
    if st.button("🩺 Get Diagnosis", type="primary", use_container_width=True):
        pred, prob = predict(user_input, model, scaler, feature_names)
        diagnosis = "Malignant" if pred == 1 else "Benign"
        if pred == 1:
            result_placeholder.error(f"### ⚠️ Prediction: {diagnosis}")
        else:
            result_placeholder.success(f"### ✅ Prediction: {diagnosis}")
        confidence_placeholder.metric("Confidence", f"{prob:.1%}")
    else:
        result_placeholder.info("Click the button above to see prediction.")
        confidence_placeholder.empty()
    
    # ---------- Team & project info (footer) ----------
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; padding: 0.5rem; background: #4b4b4b; border-radius: 12px;'>
            <strong>Group 2 Members</strong><br>
            JONATHAN NII ADARMAH SACKEY ABS24A00016Y | GETRUDE NAA MARTEY ABS24A00022Y | JASON EMMANUEL NEEQUAYE ABS24A00013Y | Emmanuel Odame Nyarko<br>
            <em>Course Project – Machine Learning for Healthcare</em><br>
            ⚠️ Educational demo only – not for clinical use.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()