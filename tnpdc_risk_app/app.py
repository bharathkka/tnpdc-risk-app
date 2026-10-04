import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
import joblib

st.set_page_config(page_title="TNPDC Event Risk Predictor", page_icon="⚡", layout="wide")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "event_risk_model.joblib")
    le = joblib.load(BASE_DIR / "label_encoder.joblib")
    feature_cols = joblib.load(BASE_DIR / "feature_columns.joblib")
    return model, le, feature_cols

model, le, feature_cols = load_model()

# ------------------ Title ------------------
st.title("⚡ TNPDC Smart Meter – Event Risk Predictor")
st.markdown("Predict whether a meter is **Critical / High / Medium / Low** risk based on Event data")

# ------------------ Sidebar ------------------
st.sidebar.header("About")
st.sidebar.info("""
This model was trained on Event-based risk logic using:
- Earth Loading
- Neutral Disturbance
- CT Bypass / Reverse / Open
- Magnetic & Cover Open events
""")

# ------------------ Input Method ------------------
option = st.radio("Select Input Method:", ["Single Meter Prediction", "Bulk Prediction (CSV Upload)"], horizontal=True)

# ============================================================
# SINGLE METER
# ============================================================
if option == "Single Meter Prediction":
    st.subheader("Enter Event Counts for a Meter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        meter_id = st.text_input("Meter Serial No", value="")
        earth = st.number_input("Earth Loading Count", min_value=0, value=0)
        neutral = st.number_input("Neutral Disturbance Count", min_value=0, value=0)
        bypass = st.number_input("CT Bypass Count", min_value=0, value=0)
    
    with col2:
        reverse = st.number_input("CT Reverse Count", min_value=0, value=0)
        open_cnt = st.number_input("CT Open Count", min_value=0, value=0)
        unbalance = st.number_input("Current Unbalance Occ Count", min_value=0, value=0)
        magnetic = st.number_input("Magnetic Tamper Count", min_value=0, value=0)
    
    with col3:
        cover = st.number_input("Cover Open Count", min_value=0, value=0)
        total_events = st.number_input("Total Events", min_value=0, value=0)
        critical_events = st.number_input("Critical Events", min_value=0, value=0)
        max_tamper = st.number_input("Max Tamper Count", min_value=0, value=0)
    
    if st.button("Predict Risk", type="primary"):
        input_data = pd.DataFrame([{
            'earth_loading_cnt': earth,
            'neutral_dist_cnt': neutral,
            'ct_bypass_cnt': bypass,
            'ct_reverse_cnt': reverse,
            'ct_open_cnt': open_cnt,
            'cnt_Current_Unbalance_Occ': unbalance,
            'cnt_Magnetic_Occ': magnetic,
            'cnt_Cover_Open': cover,
            'total_events': total_events,
            'critical_events': critical_events,
            'max_tamper_count': max_tamper
        }])
        
        # Predict
        pred_encoded = model.predict(input_data[feature_cols])
        pred_band = le.inverse_transform(pred_encoded)[0]
        proba = model.predict_proba(input_data[feature_cols])[0]
        
        st.divider()
        st.subheader(f"Result for Meter: `{meter_id if meter_id else 'Unknown'}`")
        
        # Color based on band
        if pred_band == "Critical":
            st.error(f"### Risk Band: {pred_band}")
        elif pred_band == "High":
            st.warning(f"### Risk Band: {pred_band}")
        elif pred_band == "Medium":
            st.info(f"### Risk Band: {pred_band}")
        else:
            st.success(f"### Risk Band: {pred_band}")
        
        # Probabilities
        st.write("### Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Risk Band": le.classes_,
            "Probability": proba.round(4)
        }).sort_values("Probability", ascending=False)
        st.dataframe(prob_df, use_container_width=True)

# ============================================================
# BULK PREDICTION
# ============================================================
else:
    st.subheader("Upload CSV for Bulk Prediction")
    st.markdown("""
    **Required columns:**  
    `earth_loading_cnt, neutral_dist_cnt, ct_bypass_cnt, ct_reverse_cnt, ct_open_cnt, 
    cnt_Current_Unbalance_Occ, cnt_Magnetic_Occ, cnt_Cover_Open, 
    total_events, critical_events, max_tamper_count`
    
    Optional: `MeterSerialNo`
    """)
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded: **{df.shape[0]}** meters")
        
        # Check missing columns
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("Run Bulk Prediction", type="primary"):
                with st.spinner("Predicting..."):
                    X = df[feature_cols].fillna(0)
                    preds = model.predict(X)
                    bands = le.inverse_transform(preds)
                    probas = model.predict_proba(X)
                    
                    result = df.copy()
                    result["Predicted_Risk_Band"] = bands
                    result["Prob_Critical"] = probas[:, list(le.classes_).index("Critical")].round(4)
                    result["Prob_High"] = probas[:, list(le.classes_).index("High")].round(4)
                    
                    st.success("Prediction completed!")
                    
                    # Summary
                    st.write("### Risk Band Summary")
                    st.dataframe(result["Predicted_Risk_Band"].value_counts().reset_index())
                    
                    # Show results
                    st.write("### Predictions")
                    st.dataframe(result.sort_values("Prob_Critical", ascending=False), use_container_width=True)
                    
                    # Download
                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions CSV",
                        csv,
                        "event_risk_predictions.csv",
                        "text/csv"
                    )