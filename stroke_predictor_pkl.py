"""
Robust, Streamlit-friendly model loader and inference helpers.

- No model load at import time (lazy + cached with st.cache_resource).
- Clear diagnostics printed to Streamlit Cloud logs on failure (Python/sklearn versions, tracebacks).
- Safe feature alignment using model.feature_names_in_ when available, otherwise a default order.
- Heavy/optional libs (matplotlib, seaborn, sklearn.metrics) are imported only inside plotting.
- MODEL_FILE can be overridden via env var MODEL_FILE; default is 'strokerisk_tune_ensemble_model.pkl'
  placed next to this file.
"""

from pathlib import Path
from io import BytesIO
import os
import sys
import traceback

import numpy as np
import pandas as pd
import joblib

# ------------------------------
# Optional Streamlit-aware caching (fallback if not running under Streamlit)
# ------------------------------
try:
    import streamlit as st
    _cache_resource = st.cache_resource
except Exception:
    st = None
    def _cache_resource(func):
        _cached = {}
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key not in _cached:
                _cached[key] = func(*args, **kwargs)
            return _cached[key]
        return wrapper

# ------------------------------
# File locations (override via env if needed)
# ------------------------------
CURRENT_DIR = Path(__file__).parent
# DEFAULT_MODEL_NAME = "strokerisk_tune_ensemble_model.pkl"
DEFAULT_MODEL_NAME = "strokerisk_model_rf.pkl"
MODEL_FILE = Path(os.getenv("MODEL_FILE", CURRENT_DIR / DEFAULT_MODEL_NAME))
DATA_SAMPLE_PATH = CURRENT_DIR / "data" / "sample_data.csv"

# ------------------------------
# Defaults for preprocessing (from your original stats)
# ------------------------------
age_mean, age_std = 43.23, 22.61
glucose_mean, glucose_std = 106.15, 45.28
bmi_mean, bmi_std = 28.89, 7.85
target_names = ["Low Risk", "High Risk"]

# Baseline feature order if the model doesn't expose names
DEFAULT_FEATURE_ORDER = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other', 'ever_married_Yes',
    'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children',
    'Residence_type_Urban', 'smoking_status_formerly smoked',
    'smoking_status_never smoked', 'smoking_status_smokes',
    'age_group_19-30', 'age_group_31-45', 'age_group_46-60',
    'age_group_61-75', 'age_group_76+'
]

# ------------------------------
# Model loading (lazy + cached)
# ------------------------------
@_cache_resource
def get_model():
    """
    Lazy-load and cache the model.
    Returns:
        (model, feature_columns_or_None)
    Raises:
        Exception if the model cannot be loaded.
    """
    print("=== Model load starting ===")
    print("Python:", sys.version)
    try:
        import sklearn
        print("scikit-learn:", sklearn.__version__)
    except Exception:
        print("scikit-learn: NOT IMPORTABLE")

    # Verify model exists
    if not Path(MODEL_FILE).exists():
        msg = f"Model file not found at: {MODEL_FILE}"
        print("[ERROR]", msg)
        raise FileNotFoundError(msg)

    # Load (handle PyCaret-linked pickles gracefully)
    try:
        obj = joblib.load(MODEL_FILE)
    except ModuleNotFoundError as e:
        # Common when the pickle references pycaret.* on Cloud where pycaret isn't installed
        print("[ERROR] joblib.load failed (missing module):", str(e))
        print("Hint: If this pickle was created with PyCaret, convert it to a pure scikit-learn "
              "Pipeline locally, or re-export the model with only sklearn components.")
        traceback.print_exc()
        raise
    except Exception as e:
        print("[ERROR] joblib.load failed:", str(e))
        traceback.print_exc()
        raise

    model = None
    feature_columns = None

    if isinstance(obj, dict):
        model = obj.get("model", None)
        feature_columns = obj.get("feature_columns", None)
        if model is None:
            # fallback to first estimator-like value
            for v in obj.values():
                if hasattr(v, "predict"):
                    model = v
                    break
    else:
        model = obj

    if model is None or not hasattr(model, "predict"):
        raise ValueError("Loaded object is not a valid scikit-learn estimator/pipeline")

    # Prefer feature names from the estimator if available
    try:
        if hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)
            print("Feature columns discovered on model:", len(feature_columns))
        else:
            print("Model has no feature_names_in_; will use DEFAULT_FEATURE_ORDER.")
    except Exception:
        print("Could not read feature_names_in_; will use DEFAULT_FEATURE_ORDER.")

    print("=== Model load complete ===")
    return model, feature_columns

# ------------------------------
# Validation / preprocessing
# ------------------------------
def validate_input(input_data: dict) -> bool:
    """Validate input data before processing."""
    errors = []

    required_fields = ['age', 'hypertension', 'heart_disease',
                       'avg_glucose_level', 'bmi', 'gender']
    for field in required_fields:
        if field not in input_data:
            errors.append(f"Missing required field: {field}")

    if 'age' in input_data and not (0 <= input_data['age'] <= 120):
        errors.append("Age must be between 0-120")
    if 'bmi' in input_data and not (10 <= input_data['bmi'] <= 50):
        errors.append("BMI must be between 10-50")
    if 'avg_glucose_level' in input_data and not (50 <= input_data['avg_glucose_level'] <= 300):
        errors.append("Glucose level must be between 50-300 mg/dL")

    valid_genders = ['Male', 'Female', 'Other']
    if 'gender' in input_data and input_data['gender'] not in valid_genders:
        errors.append(f"Gender must be one of: {', '.join(valid_genders)}")

    if errors:
        raise ValueError(" | ".join(errors))
    return True


def preprocess_input(input_data: dict) -> dict:
    """
    Convert frontend input to model-ready dict with one-hot columns
    matching DEFAULT_FEATURE_ORDER. Numerical fields are standardized.
    """
    processed = {
        'age': 0,
        'hypertension': 0,
        'heart_disease': 0,
        'avg_glucose_level': 0,
        'bmi': 0,
        'gender_Male': 0,
        'gender_Other': 0,
        'ever_married_Yes': 0,
        'work_type_Never_worked': 0,
        'work_type_Private': 0,
        'work_type_Self-employed': 0,
        'work_type_children': 0,
        'Residence_type_Urban': 0,
        'smoking_status_formerly smoked': 0,
        'smoking_status_never smoked': 0,
        'smoking_status_smokes': 0,
        'age_group_19-30': 0,
        'age_group_31-45': 0,
        'age_group_46-60': 0,
        'age_group_61-75': 0,
        'age_group_76+': 0
    }

    # Standardize numerics
    if 'age' in input_data:
        processed['age'] = (input_data['age'] - age_mean) / age_std
    if 'avg_glucose_level' in input_data:
        processed['avg_glucose_level'] = (input_data['avg_glucose_level'] - glucose_mean) / glucose_std
    if 'bmi' in input_data:
        processed['bmi'] = (input_data['bmi'] - bmi_mean) / bmi_std

    # Binary
    processed['hypertension'] = 1 if input_data.get('hypertension', 0) in (1, "1", True, "Yes") else 0
    processed['heart_disease'] = 1 if input_data.get('heart_disease', 0) in (1, "1", True, "Yes") else 0

    # Categorical one-hot
    gender = input_data.get('gender', 'Female')
    if gender == 'Male':
        processed['gender_Male'] = 1
    elif gender == 'Other':
        processed['gender_Other'] = 1  # <-- fixed line

    if input_data.get('ever_married', 'No') == 'Yes':
        processed['ever_married_Yes'] = 1

    work_type = input_data.get('work_type', 'Private')
    if work_type == 'Never_worked':
        processed['work_type_Never_worked'] = 1
    elif work_type == 'Private':
        processed['work_type_Private'] = 1
    elif work_type == 'Self-employed':
        processed['work_type_Self-employed'] = 1
    elif work_type == 'children':
        processed['work_type_children'] = 1

    if input_data.get('Residence_type', 'Urban') == 'Urban':
        processed['Residence_type_Urban'] = 1

    smoking_status = input_data.get('smoking_status', 'never smoked')
    if smoking_status == 'formerly smoked':
        processed['smoking_status_formerly smoked'] = 1
    elif smoking_status == 'never smoked':
        processed['smoking_status_never smoked'] = 1
    elif smoking_status == 'smokes':
        processed['smoking_status_smokes'] = 1

    # Mutually exclusive age groups
    age = input_data.get('age', 0)
    if age <= 30:
        processed['age_group_19-30'] = 1
    elif age <= 45:
        processed['age_group_31-45'] = 1
    elif age <= 60:
        processed['age_group_46-60'] = 1
    elif age <= 75:
        processed['age_group_61-75'] = 1
    else:
        processed['age_group_76+'] = 1

    return processed


def _align_features(df: pd.DataFrame, feature_columns):
    """
    Align df columns to model.feature_names_in_ if provided; otherwise to DEFAULT_FEATURE_ORDER.
    Any missing columns will be added with zeros; extras will be dropped.
    """
    if feature_columns and len(feature_columns) > 0:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    else:
        for col in DEFAULT_FEATURE_ORDER:
            if col not in df.columns:
                df[col] = 0
        df = df[DEFAULT_FEATURE_ORDER]
    return df


def _predict_proba_safe(model, X: pd.DataFrame):
    """Return (pred_label, proba_1) even if estimator lacks predict_proba."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            proba_1 = float(p[0, 1])
            pred = int(np.argmax(p[0]))
        else:
            proba_1 = float(p[0])
            pred = int(round(proba_1))
        return pred, proba_1

    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        proba_1 = float(1 / (1 + np.exp(-s[0])))
        pred = int(proba_1 >= 0.5)
        return pred, proba_1

    pred = int(model.predict(X)[0])
    proba_1 = float(pred)
    return pred, proba_1


def predict_stroke_risk(input_data: dict) -> dict:
    """Validate, preprocess, align features, and predict. Returns a dict with status."""
    try:
        validate_input(input_data)
        processed = preprocess_input(input_data)

        df = pd.DataFrame([processed])
        model, feature_columns = get_model()
        X = _align_features(df, feature_columns)

        pred, proba_1 = _predict_proba_safe(model, X)

        return {
            "status": "success",
            "prediction": pred,
            "probabilities": [1 - proba_1, proba_1],
            "risk_level": "High Risk" if pred == 1 else "Low Risk",
            "probability_percent": f"{proba_1*100:.1f}%",
            "probability_raw": proba_1
        }
    except Exception as e:
        print("[predict_stroke_risk] ERROR:", str(e))
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "risk_level": "Error",
            "probability_percent": "0.0%",
            "probabilities": [0.0, 0.0],
            "probability_raw": 0.0
        }

# ------------------------------
# Extra utilities (optional)
# ------------------------------
def get_feature_importance(top_n: int = 10) -> pd.DataFrame:
    """Return top-n feature importances if available."""
    model, feature_columns = get_model()
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model doesn't support feature importance")

    importance = np.asarray(model.feature_importances_)
    if feature_columns and len(feature_columns) == len(importance):
        names = feature_columns
    else:
        names = DEFAULT_FEATURE_ORDER[: len(importance)]

    idx = np.argsort(importance)[-top_n:][::-1]
    return pd.DataFrame({
        "Feature": [names[i] for i in idx],
        "Importance": importance[idx]
    })


def generate_what_if_scenario(base_data: dict, changes: dict) -> dict:
    """Simple what-if analysis by modifying inputs and re-predicting."""
    try:
        modified = dict(base_data)
        modified.update(changes)
        result = predict_stroke_risk(modified)
        if result.get("status") != "success":
            raise ValueError(result.get("error", "Prediction failed"))
        return {
            "status": "success",
            "original_risk": base_data.get("probability_raw", 0.0),
            "new_risk": result["probability_raw"],
            "risk_change": result["probability_raw"] - base_data.get("probability_raw", 0.0),
            "details": result
        }
    except Exception as e:
        print("[generate_what_if_scenario] ERROR:", str(e))
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def plot_model_performance():
    """
    Generate ROC curve and Confusion Matrix images from DATA_SAMPLE_PATH CSV.
    Returns a dict with raw PNG bytes for 'roc_curve' and 'confusion_matrix'.
    """
    try:
        # Lazy imports to avoid import-time crashes if optional deps are missing
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc, confusion_matrix

        if not DATA_SAMPLE_PATH.exists():
            raise FileNotFoundError(f"Sample data not found at {DATA_SAMPLE_PATH}")

        df = pd.read_csv(DATA_SAMPLE_PATH)
        if "stroke" not in df.columns:
            raise ValueError("Sample data must contain a 'stroke' target column")

        y = df["stroke"].astype(int)
        X = df.drop(columns=["stroke"])

        model, feature_columns = get_model()
        X = _align_features(X, feature_columns)

        # ROC
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            s = model.decision_function(X)
            y_scores = 1 / (1 + np.exp(-s))
        else:
            y_scores = model.predict(X).astype(float)

        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_buf = BytesIO()
        plt.savefig(roc_buf, format='png', bbox_inches='tight')
        plt.close()

        # Confusion matrix
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_buf = BytesIO()
        plt.savefig(cm_buf, format='png', bbox_inches='tight')
        plt.close()

        return {"roc_curve": roc_buf.getvalue(), "confusion_matrix": cm_buf.getvalue()}

    except Exception as e:
        print("[plot_model_performance] ERROR:", str(e))
        traceback.print_exc()
        raise ValueError(f"Performance visualization failed: {str(e)}")


def get_feature_ranges():
    return {
        'age': {'min': 0, 'max': 120, 'mean': age_mean, 'std': age_std},
        'bmi': {'min': 10, 'max': 50, 'mean': bmi_mean, 'std': bmi_std},
        'avg_glucose_level': {'min': 50, 'max': 300, 'mean': glucose_mean, 'std': glucose_std}
    }


def get_categorical_options():
    return {
        'gender': ['Male', 'Female', 'Other'],
        'hypertension': ['No', 'Yes'],
        'heart_disease': ['No', 'Yes'],
        'ever_married': ['No', 'Yes'],
        'work_type': ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
        'Residence_type': ['Urban', 'Rural'],
        'smoking_status': ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
    }
