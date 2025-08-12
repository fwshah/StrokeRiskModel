import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# --- Model Loading System ---
@st.cache_resource
def load_models():
    """Load all individual models for analysis"""
    try:
        model_dir = Path(__file__).resolve().parent.parent

        return {
            "Random Forest": joblib.load(model_dir / "strokerisk_model_rf.pkl"),
            "XGBoost": joblib.load(model_dir / "strokerisk_model_xgboost.pkl"),
            "Extra Trees": joblib.load(model_dir / "strokerisk_model_et.pkl")
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# --- Complete Feature Processing ---
def preprocess_input(input_data):
    """Prepare input matching the original training data features"""
    processed = {
        # Numerical features (standardized)
        'age': float((input_data['age'] - 43.23) / 22.61),
        'avg_glucose_level': float((input_data['avg_glucose_level'] - 106.15) / 45.28),
        'bmi': float((input_data['bmi'] - 28.89) / 7.85),
        
        # Binary features
        'hypertension': int(input_data.get('hypertension', 0)),
        'heart_disease': int(input_data.get('heart_disease', 0)),
        
        # Gender (only Male/Other as per training data)
        'gender_Male': int(input_data.get('gender') == 'Male'),
        'gender_Other': int(input_data.get('gender') == 'Other'),
        
        # Marriage status (only Yes as per training data)
        'ever_married_Yes': int(input_data.get('ever_married') == 'Yes'),
        
        # Work type (excluding Govt_job)
        'work_type_Never_worked': int(input_data.get('work_type') == 'Never_worked'),
        'work_type_Private': int(input_data.get('work_type') == 'Private'),
        'work_type_Self-employed': int(input_data.get('work_type') == 'Self-employed'),
        'work_type_children': int(input_data.get('work_type') == 'children'),
        
        # Residence type (only Urban)
        'Residence_type_Urban': int(input_data.get('Residence_type') == 'Urban'),
        
        # Smoking status (excluding Unknown)
        'smoking_status_formerly smoked': int(input_data.get('smoking_status') == 'formerly smoked'),
        'smoking_status_never smoked': int(input_data.get('smoking_status') == 'never smoked'),
        'smoking_status_smokes': int(input_data.get('smoking_status') == 'smokes'),
        
        # Age groups
        'age_group_19-30': int(19 <= input_data['age'] <= 30),
        'age_group_31-45': int(31 <= input_data['age'] <= 45),
        'age_group_46-60': int(46 <= input_data['age'] <= 60),
        'age_group_61-75': int(61 <= input_data['age'] <= 75),
        'age_group_76+': int(input_data['age'] > 75)
    }
    
    # Ensure all features are in correct order
    feature_columns = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Male', 'gender_Other',
        'ever_married_Yes',
        'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children',
        'Residence_type_Urban',
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
        'age_group_19-30', 'age_group_31-45', 'age_group_46-60',
        'age_group_61-75', 'age_group_76+'
    ]
    
    return pd.DataFrame([processed], columns=feature_columns)

# --- Analysis Page ---
def risk_analysis_engine():
    st.set_page_config(layout="wide")
    st.title("🔬 Stroke Risk Analysis Engine")
    
    # Check for patient data
    if "patient_data" not in st.session_state:
        st.warning("No patient data found. Complete the assessment first.")
        if st.button("Go to Assessment"):
            st.switch_page("pages/02_Patient_Data_Entry.py")
        return

    # Load models and data
    models = load_models()
    input_data = st.session_state["patient_data"]
    processed_data = preprocess_input(input_data)
    
    # --- Model Predictions ---
    results = {}
    for name, model in models.items():
        try:
            proba = model.predict_proba(processed_data)[0]
            results[name] = {
                "Low Risk": float(proba[0]),
                "High Risk": float(proba[1]),
                "Features": {k: float(v) for k, v in zip(processed_data.columns, model.feature_importances_)}
            }
        except Exception as e:
            st.error(f"{name} model failed: {str(e)}")
            continue
    
    if not results:
        st.error("All models failed to predict. Check model compatibility.")
        return
    
    # Summary Section
    with st.expander("📊 Risk Summary", expanded=True):
        avg_risk = np.mean([v["High Risk"] for v in results.values()]) * 100
        risk_color = "#FF4B4B" if avg_risk > 30 else ("#F9D423" if avg_risk > 15 else "#09AB3B")
        st.markdown(f"""
        <div style="background-color:{risk_color};padding:10px;border-radius:5px;color:white;text-align:center">
            <h2 style="color:white;">Average Stroke Risk: {avg_risk:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Risk Probability")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            for model in results:
                ax1.bar(model, results[model]["High Risk"], color='#FF6B6B')
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Probability")
            plt.xticks(rotation=45)
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Model Agreement")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            pd.DataFrame(results).T[["Low Risk", "High Risk"]].plot(
                kind='bar', stacked=True, ax=ax2, 
                color=['#4ECDC4', '#FF6B6B'])
            ax2.set_ylabel("Probability")
            st.pyplot(fig2)
    
    # --- Feature Importance ---
    with st.expander("🔍 Feature Analysis"):
        tabs = st.tabs([f"{name}" for name in results.keys()])
        
        for (name, model_data), tab in zip(results.items(), tabs):
            with tab:
                importance = pd.DataFrame({
                    "Feature": model_data["Features"].keys(),
                    "Importance": [float(v) for v in model_data["Features"].values()]
                }).sort_values("Importance", ascending=False)
                
                st.subheader(f"{name} Key Drivers")
                
                # Top 10 features visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                importance.head(10).plot.barh(x="Feature", y="Importance", ax=ax, color='#1F77B4')
                ax.set_title(f"Top 10 Predictive Features ({name})")
                st.pyplot(fig)
                
                # Full feature table
                importance["Importance"] = importance["Importance"].astype(float)
                st.dataframe(
                    importance.style.background_gradient(cmap='Blues'),
                    height=400,
                    column_config={
                        "Feature": "Risk Factor",
                        "Importance": st.column_config.ProgressColumn(
                            "Importance",
                            format="%.3f",
                            min_value=0.0,
                            max_value=float(importance["Importance"].max())
                        )
                    }
                )
    
    # --- Clinical Interpretation ---
    with st.expander("💡 Clinical Insights & Recommendations"):
        risk_factors = []
        protective_factors = []
        
        # Identify key factors
        if input_data['age'] > 60:
            risk_factors.append(f"Age ({input_data['age']} years)")
        if input_data['hypertension']:
            risk_factors.append("Hypertension")
        if input_data['avg_glucose_level'] > 140:
            risk_factors.append(f"High glucose ({input_data['avg_glucose_level']:.1f} mg/dL)")
        if input_data['smoking_status'] in ['formerly smoked', 'smokes']:
            risk_factors.append("Smoking history")
        
        if input_data['age'] < 40:
            protective_factors.append("Younger age")
        if not input_data['hypertension']:
            protective_factors.append("No hypertension")
        if input_data['smoking_status'] == 'never smoked':
            protective_factors.append("Never smoked")
        
        st.write(f"""
        - **Consensus Risk**: {avg_risk:.1f}% average across models
        - **Key Risk Factors**: {', '.join(risk_factors) if risk_factors else "None identified"}
        - **Protective Factors**: {', '.join(protective_factors) if protective_factors else "None identified"}
        """)
        
        # Recommendations
        st.subheader("Recommendations")
        if avg_risk > 30:
            st.warning("""
            🚨 High Risk Detected:
            - Consult a cardiovascular specialist
            - Consider lifestyle interventions
            - Monitor blood pressure regularly
            """)
        elif avg_risk > 15:
            st.info("""
            ⚠️ Moderate Risk Detected:
            - Annual stroke risk screening recommended
            - Maintain healthy diet and exercise
            """)
        else:
            st.success("""
            ✅ Low Risk Detected:
            - Continue preventive health measures
            - Regular check-ups recommended
            """)

if __name__ == "__main__":
    risk_analysis_engine()