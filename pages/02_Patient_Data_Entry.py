# import streamlit as st
# import pandas as pd
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# try:
#     from stroke_predictor_pkl import predict_stroke_risk
# except ModuleNotFoundError:
#     st.error("Could not import 'stroke_predictor_pkl'. Please ensure the file exists in the parent directory and is named correctly.")
#     def predict_stroke_risk(*args, **kwargs):
#         raise ImportError("stroke_predictor_pkl module not found.")

# def patient_data_entry():
#     # st.set_page_config(page_title="Patient Data Entry", layout="wide")
#     st.set_page_config(page_title="Patient Data Entry", layout="wide")  # first Streamlit call

#     st.title("🩺 Stroke Risk Assessment System")
    
#     # Initialize all required session state variables
#     if 'risk_result' not in st.session_state:
#         st.session_state.risk_result = None
#         st.session_state.patient_data = None
#         st.session_state.patient_id = None
    
#     # Main form
#     with st.form("patient_form"):
#         st.subheader("Patient Identification")
#         patient_id = st.text_input("Patient ID/Name", placeholder="PT-001")
        
#         st.subheader("Medical Information")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             age = st.number_input("Age", min_value=0, max_value=120, value=50)
#             hypertension = st.radio("Hypertension", ["No", "Yes"], index=0)
#             heart_disease = st.radio("Heart Disease", ["No", "Yes"], index=0)
#             avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", 
#                                             min_value=50.0, max_value=300.0, value=100.0)
#             bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        
#         with col2:
#             gender = st.radio("Gender", ["Male", "Female", "Other"], index=1)
#             ever_married = st.radio("Ever Married", ["No", "Yes"], index=1)
#             work_type = st.selectbox("Work Type", 
#                                 ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
#                                 index=0)
#             residence_type = st.radio("Residence Type", ["Urban", "Rural"], index=0)
#             smoking_status = st.selectbox("Smoking Status", 
#                                     ["never smoked", "formerly smoked", "smokes", "Unknown"],
#                                     index=0)
        
#         submitted = st.form_submit_button("Assess Stroke Risk", type="primary")
    
#     if submitted:
#         # Store all data in session state
#         st.session_state.patient_id = patient_id
#         patient_data = {
#             'age': age,
#             'hypertension': 1 if hypertension == "Yes" else 0,
#             'heart_disease': 1 if heart_disease == "Yes" else 0,
#             'avg_glucose_level': avg_glucose_level,
#             'bmi': bmi,
#             'gender': gender,
#             'ever_married': ever_married,
#             'work_type': work_type,
#             'Residence_type': residence_type,
#             'smoking_status': smoking_status
#         }
#         st.session_state.patient_data = patient_data
        
#         try:
#             with st.spinner("Calculating stroke risk..."):
#                 result = predict_stroke_risk(patient_data)
                
#                 if result.get('status') != 'success':
#                     raise ValueError(result.get('error', 'Prediction failed'))
                
#                 # Store complete results
#                 st.session_state.risk_result = {
#                     'patient_id': patient_id,
#                     'risk_level': result['risk_level'],
#                     'probability_percent': result['probability_percent'],
#                     'probabilities': result['probabilities'],
#                     'probability_raw': result['probability_raw'],
#                     'input_data': patient_data
#                 }
                
#                 st.success("Assessment completed!")
#                 st.balloons()
                
#         except Exception as e:
#             st.error(f"Assessment failed: {str(e)}")
#             st.session_state.risk_result = None
    
#     # Display results if available
#     if st.session_state.get('risk_result'):
#         display_results()

# def display_results():
#     result = st.session_state.risk_result
#     input_data = result['input_data']
    
#     st.divider()
#     st.subheader(f"Assessment Results for {result['patient_id']}")
    
#     # Risk level display
#     risk_color = "#FF4B4B" if result['risk_level'] == "High Risk" else "#09AB3B"
#     st.markdown(f"""
#     <div style="background-color:{risk_color};padding:10px;border-radius:5px;color:white;text-align:center">
#         <h2 style="color:white;">{result['risk_level']} ({result['probability_percent']})</h2>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Probability breakdown
#     st.markdown("#### Probability Breakdown")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Low Risk", f"{result['probabilities'][0]*100:.1f}%")
#     with col2:
#         st.metric("High Risk", f"{result['probabilities'][1]*100:.1f}%")
    
#     # What-if analysis
#     st.divider()
#     st.subheader("Scenario Analysis")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         new_age = st.slider("Adjust Age", 1, 120, int(input_data['age']), key="what_if_age")
#     with col2:
#         new_bmi = st.slider("Adjust BMI", 10.0, 50.0, float(input_data['bmi']), key="what_if_bmi")
    
#     if st.button("Run Scenario", key="scenario_button"):
#         try:
#             with st.spinner("Calculating..."):
#                 modified_data = input_data.copy()
#                 modified_data.update({'age': new_age, 'bmi': new_bmi})
                
#                 new_result = predict_stroke_risk(modified_data)
#                 if new_result.get('status') != 'success':
#                     raise ValueError(new_result.get('error', 'Scenario analysis failed'))
                
#                 # Display comparison
#                 st.markdown("#### Scenario Results")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Original Risk", result['probability_percent'])
#                 with col2:
#                     change = (new_result['probability_raw'] - result['probability_raw']) * 100
#                     st.metric("New Risk", new_result['probability_percent'],
#                             delta=f"{change:+.1f}%")
                
#                 if change > 5:  # Significant change threshold
#                     st.warning(f"Major risk change detected ({change:+.1f}%)")
                
#         except Exception as e:
#             st.error(f"Scenario analysis failed: {str(e)}")
    
#     # Advanced Analysis Button (only shown after initial assessment)
#     st.divider()
#     if st.button("🔍 Launch Advanced Analysis", type="primary", 
#                 help="Detailed multi-model analysis"):
#         st.session_state.analysis_ready = True
#         st.switch_page("pages/03_Risk_Assessment_Results.py")

# if __name__ == "__main__":
#     patient_data_entry()

# pages/02_Patient_Data_Entry.py

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# --- 1) Page config: must be the first Streamlit call in this script
st.set_page_config(page_title="Patient Data Entry", layout="wide")

# --- 2) Ensure we can import from the app root (parent of /pages)
PAGES_DIR = Path(__file__).resolve().parent
APP_ROOT = PAGES_DIR.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# --- 3) Import predictor and (optionally) warm up model load
# We try to import both predict_stroke_risk and get_model (if available)
try:
    from stroke_predictor_pkl import predict_stroke_risk  # required
    try:
        # Optional but recommended: warm model to surface errors in logs immediately
        from stroke_predictor_pkl import get_model
        _ = get_model()  # triggers a single load; failures will print details to logs
    except Exception:
        # If get_model isn't available or fails, we continue; prediction path will handle errors.
        pass
except ModuleNotFoundError:
    st.error("Could not import 'stroke_predictor_pkl'. "
             "Ensure the file exists at the app root and is named exactly 'stroke_predictor_pkl.py'.")
    st.stop()

# -----------------------------
# UI / App Logic
# -----------------------------
def patient_data_entry():
    st.title("🩺 Stroke Risk Assessment System")

    # Initialize session state keys
    for key, default in [
        ("risk_result", None),
        ("patient_data", None),
        ("patient_id", None),
        ("analysis_ready", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ---- Patient form ----
    with st.form("patient_form"):
        st.subheader("Patient Identification")
        patient_id = st.text_input("Patient ID/Name", placeholder="PT-001")

        st.subheader("Medical Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            hypertension = st.radio("Hypertension", ["No", "Yes"], index=0)
            heart_disease = st.radio("Heart Disease", ["No", "Yes"], index=0)
            avg_glucose_level = st.number_input(
                "Average Glucose Level (mg/dL)",
                min_value=50.0, max_value=300.0, value=100.0
            )
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

        with col2:
            gender = st.radio("Gender", ["Male", "Female", "Other"], index=1)
            ever_married = st.radio("Ever Married", ["No", "Yes"], index=1)
            work_type = st.selectbox(
                "Work Type",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                index=0
            )
            residence_type = st.radio("Residence Type", ["Urban", "Rural"], index=0)
            smoking_status = st.selectbox(
                "Smoking Status",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                index=0
            )

        submitted = st.form_submit_button("Assess Stroke Risk", type="primary")

    if submitted:
        # Basic validation to avoid empty patient IDs if you need them
        if not patient_id.strip():
            st.warning("Please enter a Patient ID/Name.")
            st.stop()

        # Store all data in session state
        st.session_state.patient_id = patient_id
        patient_data = {
            'age': age,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'avg_glucose_level': float(avg_glucose_level),
            'bmi': float(bmi),
            'gender': gender,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'smoking_status': smoking_status
        }
        st.session_state.patient_data = patient_data

        try:
            with st.spinner("Calculating stroke risk..."):
                result = predict_stroke_risk(patient_data)

            if result.get('status') != 'success':
                raise ValueError(result.get('error', 'Prediction failed'))

            # Store results
            st.session_state.risk_result = {
                'patient_id': patient_id,
                'risk_level': result['risk_level'],
                'probability_percent': result['probability_percent'],
                'probabilities': result['probabilities'],
                'probability_raw': result['probability_raw'],
                'input_data': patient_data
            }

            st.success("Assessment completed!")
            try:
                st.balloons()
            except Exception:
                pass

        except Exception as e:
            st.error(f"Assessment failed: {str(e)}")
            st.session_state.risk_result = None

    # Display results if available
    if st.session_state.get('risk_result'):
        display_results()


def display_results():
    result = st.session_state.risk_result
    input_data = result['input_data']

    st.divider()
    st.subheader(f"Assessment Results for {result['patient_id']}")

    # Risk level banner
    risk_color = "#FF4B4B" if result['risk_level'] == "High Risk" else "#09AB3B"
    st.markdown(
        f"""
        <div style="background-color:{risk_color};padding:10px;border-radius:5px;color:white;text-align:center">
            <h2 style="color:white;">{result['risk_level']} ({result['probability_percent']})</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability breakdown
    st.markdown("#### Probability Breakdown")
    col1, col2 = st.columns(2)
    try:
        low = float(result['probabilities'][0]) * 100
        high = float(result['probabilities'][1]) * 100
    except Exception:
        low, high = 0.0, 0.0

    with col1:
        st.metric("Low Risk", f"{low:.1f}%")
    with col2:
        st.metric("High Risk", f"{high:.1f}%")

    # What-if analysis
    st.divider()
    st.subheader("Scenario Analysis")

    col1, col2 = st.columns(2)
    with col1:
        new_age = st.slider("Adjust Age", 1, 120, int(input_data['age']), key="what_if_age")
    with col2:
        new_bmi = st.slider("Adjust BMI", 10.0, 50.0, float(input_data['bmi']), key="what_if_bmi")

    if st.button("Run Scenario", key="scenario_button"):
        try:
            with st.spinner("Calculating..."):
                modified_data = input_data.copy()
                modified_data.update({'age': new_age, 'bmi': new_bmi})

                new_result = predict_stroke_risk(modified_data)
                if new_result.get('status') != 'success':
                    raise ValueError(new_result.get('error', 'Scenario analysis failed'))

                # Display comparison
                st.markdown("#### Scenario Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Risk", result['probability_percent'])
                with col2:
                    change = (new_result['probability_raw'] - result['probability_raw']) * 100
                    st.metric("New Risk", new_result['probability_percent'],
                              delta=f"{change:+.1f}%")

                if change > 5:  # Significant change threshold
                    st.warning(f"Major risk change detected ({change:+.1f}%)")

        except Exception as e:
            st.error(f"Scenario analysis failed: {str(e)}")

    # Advanced Analysis navigation
    st.divider()
    if st.button("🔍 Launch Advanced Analysis", type="primary",
                 help="Detailed multi-model analysis"):
        st.session_state.analysis_ready = True
        try:
            # Works on Streamlit >= 1.25
            st.switch_page("pages/03_Risk_Assessment_Results.py")
        except Exception:
            st.info("Use the sidebar to open '03_Risk_Assessment_Results' page.")


if __name__ == "__main__":
    patient_data_entry()
