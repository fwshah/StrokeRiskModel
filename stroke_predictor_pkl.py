import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# Configuration - UPDATED PATH HANDLING
current_dir = Path(__file__).parent
MODEL_PATH = current_dir / "strokerisk_tune_ensemble_model.pkl"
DATA_SAMPLE_PATH = current_dir / "data" / "sample_data.csv"

# Robust model loading with error handling
try:
    # Verify model file exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Load model data
    model_data = joblib.load(MODEL_PATH)
    
    # Handle different model saving formats
    if isinstance(model_data, dict):
        # Case 1: Model saved as dictionary with keys
        model = model_data.get("model")
        feature_columns = model_data.get("feature_columns", [])
        
        # Fallback if 'model' key doesn't exist
        if model is None:
            for value in model_data.values():
                if hasattr(value, 'predict'):
                    model = value
                    break
    else:
        # Case 2: Model saved directly
        model = model_data
        feature_columns = []
    
    # Validate loaded model
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a valid scikit-learn model")
    
    # Define feature columns if not loaded
    if not feature_columns:
        feature_columns = [
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Female', 'gender_Male', 'gender_Other',
            'ever_married_No', 'ever_married_Yes',
            'work_type_Govt_job', 'work_type_Never_worked',
            'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Rural', 'Residence_type_Urban',
            'smoking_status_Unknown', 'smoking_status_formerly smoked',
            'smoking_status_never smoked', 'smoking_status_smokes'
        ]

except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Training data stats
age_mean, age_std = 43.23, 22.61
glucose_mean, glucose_std = 106.15, 45.28
bmi_mean, bmi_std = 28.89, 7.85
target_names = ["Low Risk", "High Risk"]

def validate_input(input_data):
    """Validate input data before processing"""
    errors = []
    
    # Check required fields
    required_fields = ['age', 'hypertension', 'heart_disease', 
                    'avg_glucose_level', 'bmi', 'gender']
    for field in required_fields:
        if field not in input_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate numerical ranges
    if 'age' in input_data and not (0 <= input_data['age'] <= 120):
        errors.append("Age must be between 0-120")
    if 'bmi' in input_data and not (10 <= input_data['bmi'] <= 50):
        errors.append("BMI must be between 10-50")
    if 'avg_glucose_level' in input_data and not (50 <= input_data['avg_glucose_level'] <= 300):
        errors.append("Glucose level must be between 50-300 mg/dL")
    
    # Validate categorical values
    valid_genders = ['Male', 'Female', 'Other']
    if 'gender' in input_data and input_data['gender'] not in valid_genders:
        errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
    
    if errors:
        raise ValueError(" | ".join(errors))
    return True

def preprocess_input(input_data):
    """Convert frontend input to model-ready format with proper feature encoding"""
    # Initialize all features with 0 (using your actual feature names)
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
    
    # Handle numerical features (standardized values)
    if 'age' in input_data:
        processed['age'] = (input_data['age'] - age_mean) / age_std
    if 'avg_glucose_level' in input_data:
        processed['avg_glucose_level'] = (input_data['avg_glucose_level'] - glucose_mean) / glucose_std
    if 'bmi' in input_data:
        processed['bmi'] = (input_data['bmi'] - bmi_mean) / bmi_std
    
    # Binary features
    processed['hypertension'] = 1 if input_data.get('hypertension', 0) == 1 else 0
    processed['heart_disease'] = 1 if input_data.get('heart_disease', 0) == 1 else 0
    
    # Handle categorical features (one-hot encoding)
    gender = input_data.get('gender', 'Female')
    if gender == 'Male':
        processed['gender_Male'] = 1
    elif gender == 'Other':
        processed['gender_Other'] = 1
    
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
    
    # Handle age groups (mutually exclusive)
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

def predict_stroke_risk(input_data):
    """Final working prediction function"""
    try:
        validate_input(input_data)
        processed_data = preprocess_input(input_data)
        
        # Create dataframe with EXACTLY the right columns in right order
        df = pd.DataFrame([processed_data], columns=[
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Male', 'gender_Other', 'ever_married_Yes',
            'work_type_Never_worked', 'work_type_Private',
            'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban', 'smoking_status_formerly smoked',
            'smoking_status_never smoked', 'smoking_status_smokes',
            'age_group_19-30', 'age_group_31-45', 'age_group_46-60',
            'age_group_61-75', 'age_group_76+'
        ])
        
        # Get prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        return {
            "status": "success",
            "prediction": int(prediction),
            "probabilities": probabilities.tolist(),
            "risk_level": "High Risk" if prediction == 1 else "Low Risk",
            "probability_percent": f"{probabilities[1]*100:.1f}%",
            "probability_raw": float(probabilities[1])
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "risk_level": "Error",
            "probability_percent": "0.0%",
            "probabilities": [0.0, 0.0],
            "probability_raw": 0.0
        }

def get_feature_importance(top_n=10):
    """Get feature importance from model"""
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model doesn't support feature importance")
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:][::-1]
    
    return pd.DataFrame({
        'Feature': [feature_columns[i] for i in indices],
        'Importance': importance[indices]
    })

def generate_what_if_scenario(base_data, changes):
    """Run what-if analysis with modified features"""
    try:
        modified_data = base_data.copy()
        modified_data.update(changes)
        
        result = predict_stroke_risk(modified_data)
        if result.get('status') != 'success':
            raise ValueError(result.get('error', 'Prediction failed'))
        
        return {
            "status": "success",
            "original_risk": base_data.get('probability_raw', 0),
            "new_risk": result['probability_raw'],
            "risk_change": result['probability_raw'] - base_data.get('probability_raw', 0),
            "details": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def plot_model_performance():
    """Generate performance plots using sample data"""
    try:
        df = pd.read_csv(DATA_SAMPLE_PATH)
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        # ROC Curve
        y_scores = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_buf = BytesIO()
        plt.savefig(roc_buf, format='png')
        plt.close()
        
        # Confusion Matrix
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_buf = BytesIO()
        plt.savefig(cm_buf, format='png')
        plt.close()
        
        return {
            "roc_curve": roc_buf.getvalue(),
            "confusion_matrix": cm_buf.getvalue()
        }
    except Exception as e:
        raise ValueError(f"Performance visualization failed: {str(e)}")

def get_feature_ranges():
    """Return valid ranges for numerical features"""
    return {
        'age': {'min': 0, 'max': 120, 'mean': age_mean, 'std': age_std},
        'bmi': {'min': 10, 'max': 50, 'mean': bmi_mean, 'std': bmi_std},
        'avg_glucose_level': {'min': 50, 'max': 300, 'mean': glucose_mean, 'std': glucose_std}
    }

def get_categorical_options():
    """Return valid options for categorical features"""
    return {
        'gender': ['Male', 'Female', 'Other'],
        'hypertension': ['No', 'Yes'],
        'heart_disease': ['No', 'Yes'],
        'ever_married': ['No', 'Yes'],
        'work_type': ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
        'Residence_type': ['Urban', 'Rural'],
        'smoking_status': ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
    }