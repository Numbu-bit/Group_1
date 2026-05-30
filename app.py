import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Configure the web application layout
st.set_page_config(
    page_title="Breast Cancer Diagnostic Panel",
    page_icon="🔬",
    layout="wide",
)

# 2. Automated On-the-Fly Training Cache Setup
@st.cache_resource
def initialize_ml_pipeline():
    """Reads the raw CSV data, cleans it, trains a Random Forest classifier,
    and returns the live model along with feature averages for the UI.
    """
    # Look for data.csv in the same folder as this script (repository root)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
    
    if not os.path.exists(data_path):
        st.error(f"🚨 **Critical Error: Dataset missing at path:** `{data_path}`")
        st.info("💡 Please make sure `data.csv` is uploaded to the root of your repository.")
        st.stop()
        
    # Load and clean dataset
    df = pd.read_csv(data_path)
    df = df.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Calculate global averages to serve as dynamic baselines for the UI input fields
    feature_defaults = X.mean().to_dict()

    # Train Random Forest Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return {"model": model, "defaults": feature_defaults}

# Extract the active model and UI defaults from the live memory pipeline
try:
    pipeline = initialize_ml_pipeline()
    model = pipeline["model"]
    defaults = pipeline["defaults"]
except Exception as e:
    st.error(f"❌ Failed to initialize the machine learning pipeline: {e}")
    st.stop()

# 3. User Interface Design
st.title("🔬 Live Random Forest - Cancer Prediction Interface")
st.markdown(
    "Adjust the cellular metrics below. The model trains automatically from raw data and runs real-time inference."
)

st.markdown("""
### 👥 Project Contributors
* **Abel Archibong** — `ADS24B00176Y`
* **Anago Michael** — `ABS24A00030Y` 
* **Assitan Camara** — `ABS24A00015Y`  
---
""", unsafe_allow_html=True)                      

# Strict tracking sequence to align inputs with Scikit-learn array format
FEATURE_KEYS = list(defaults.keys())

# Segment the 30 features into category tabs to keep the UI tidy
tab1, tab2, tab3 = st.tabs(["Mean Metrics", "Standard Error Metrics", "Worst/Extreme Metrics"])
input_features = {}

# Layout Section 1: Means
with tab1:
    st.subheader("Mean Attributes of Cell Nuclei")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_features["radius_mean"] = st.number_input("Radius (Mean)", value=defaults.get("radius_mean", 14.12))
        input_features["texture_mean"] = st.number_input("Texture (Mean)", value=defaults.get("texture_mean", 19.28))
        input_features["perimeter_mean"] = st.number_input("Perimeter (Mean)", value=defaults.get("perimeter_mean", 91.96))
    with c2:
        input_features["area_mean"] = st.number_input("Area (Mean)", value=defaults.get("area_mean", 654.88))
        input_features["smoothness_mean"] = st.number_input("Smoothness (Mean)", value=defaults.get("smoothness_mean", 0.096))
        input_features["compactness_mean"] = st.number_input("Compactness (Mean)", value=defaults.get("compactness_mean", 0.104))
    with c3:
        input_features["concavity_mean"] = st.number_input("Concavity (Mean)", value=defaults.get("concavity_mean", 0.088))
        input_features["concave points_mean"] = st.number_input("Concave Points (Mean)", value=defaults.get("concave points_mean", 0.048))
        input_features["symmetry_mean"] = st.number_input("Symmetry (Mean)", value=defaults.get("symmetry_mean", 0.181))
        input_features["fractal_dimension_mean"] = st.number_input("Fractal Dimension (Mean)", value=defaults.get("fractal_dimension_mean", 0.062))

# Layout Section 2: Standard Errors
with tab2:
    st.subheader("Standard Error (SE) Attributes")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_features["radius_se"] = st.number_input("Radius SE", value=defaults.get("radius_se", 0.405))
        input_features["texture_se"] = st.number_input("Texture SE", value=defaults.get("texture_se", 1.216))
        input_features["perimeter_se"] = st.number_input("Perimeter SE", value=defaults.get("perimeter_se", 2.866))
    with c2:
        input_features["area_se"] = st.number_input("Area SE", value=defaults.get("area_se", 40.337))
        input_features["smoothness_se"] = st.number_input("Smoothness SE", value=defaults.get("smoothness_se", 0.007))
        input_features["compactness_se"] = st.number_input("Compactness SE", value=defaults.get("compactness_se", 0.025))
    with c3:
        input_features["concavity_se"] = st.number_input("Concavity SE", value=defaults.get("concavity_se", 0.031))
        input_features["concave points_se"] = st.number_input("Concave Points SE", value=defaults.get("concave points_se", 0.011))
        input_features["symmetry_se"] = st.number_input("Symmetry SE", value=defaults.get("symmetry_se", 0.020))
        input_features["fractal_dimension_se"] = st.number_input("Fractal Dimension SE", value=defaults.get("fractal_dimension_se", 0.003))

# Layout Section 3: Worst Metrics
with tab3:
    st.subheader("Worst/Largest Recorded Attributes")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_features["radius_worst"] = st.number_input("Radius (Worst)", value=defaults.get("radius_worst", 16.269))
        input_features["texture_worst"] = st.number_input("Texture (Worst)", value=defaults.get("texture_worst", 25.677))
        input_features["perimeter_worst"] = st.number_input("Perimeter (Worst)", value=defaults.get("perimeter_worst", 107.261))
    with c2:
        input_features["area_worst"] = st.number_input("Area (Worst)", value=defaults.get("area_worst", 880.583))
        input_features["smoothness_worst"] = st.number_input("Smoothness (Worst)", value=defaults.get("smoothness_worst", 0.132))
        input_features["compactness_worst"] = st.number_input("Compactness (Worst)", value=defaults.get("compactness_worst", 0.254))
    with c3:
        input_features["concavity_worst"] = st.number_input("Concavity (Worst)", value=defaults.get("concavity_worst", 0.272))
        input_features["concave points_worst"] = st.number_input("Concave Points (Worst)", value=defaults.get("concave points_worst", 0.114))
        input_features["symmetry_worst"] = st.number_input("Symmetry (Worst)", value=defaults.get("symmetry_worst", 0.290))
        input_features["fractal_dimension_worst"] = st.number_input("Fractal Dimension (Worst)", value=defaults.get("fractal_dimension_worst", 0.083))

st.write("---")

# 4. Real-Time Execution Block
if st.button("Generate Classification Diagnostic", type="primary", use_container_width=True):
    # Map user form metrics securely into a 2D mathematical vector structure matching training schema
    feature_vector = np.array([input_features[key] for key in FEATURE_KEYS]).reshape(1, -1)

    # Compute immediate real-time calculations
    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]

    st.subheader("📋 Diagnostic Results Output")
    col_banner, col_graph = st.columns([1, 1])

    with col_banner:
        if prediction == 1:
            st.error(f"### 🚨 Diagnostic Result: MALIGNANT")
            st.markdown(
                f"**Classification Confidence Score:** `{probabilities[1]*100:.2f}%`  \n"
                "The cell structure configurations align mathematically with cancerous tumors. "
                "Immediate specialist review and biopsy confirmation are strongly advised."
            )
        else:
            st.success(f"### ✅ Diagnostic Result: BENIGN")
            st.markdown(
                f"**Classification Confidence Score:** `{probabilities[0]*100:.2f}%`  \n"
                "The sample configurations indicate standard, non-cancerous cellular variations. "
                "Normal routine tracking monitoring is recommended."
            )

    with col_graph:
        # Generate an automated metrics chart using Streamlit native components
        chart_data = pd.DataFrame(
            {"Probability State (%)": probabilities * 100}, 
            index=["Benign", "Malignant"]
        )
        st.bar_chart(chart_data)
