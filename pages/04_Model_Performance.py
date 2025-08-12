import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                            recall_score, accuracy_score, confusion_matrix, 
                            classification_report, RocCurveDisplay)

# --- Model Loading System ---
@st.cache_resource
def load_models():
    """Load all individual models for analysis"""
    try:
        model_dir = Path(__file__).resolve().parent.parent

        return {
            "Ensemble": joblib.load(model_dir / "strokerisk_tune_ensemble_model.pkl"),
            "Random Forest": joblib.load(model_dir / "strokerisk_model_rf.pkl"),
            "XGBoost": joblib.load(model_dir / "strokerisk_model_xgboost.pkl"),
            "Extra Trees": joblib.load(model_dir / "strokerisk_model_et.pkl")
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# --- Model Evaluation Functions ---
def evaluate_model(model, X, y):
    """Calculate all metrics for a single model"""
    try:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba) if y_proba is not None else None,
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
    except Exception as e:
        st.error(f"Evaluation failed for {model.__class__.__name__}: {str(e)}")
        return None

def plot_roc_curves(models, X, y):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            try:
                RocCurveDisplay.from_estimator(model, X, y, name=name, ax=ax)
            except Exception as e:
                st.warning(f"Could not plot ROC for {name}: {str(e)}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('ROC Curve Comparison')
    return fig

# --- Data Loading Helper ---
def load_models_and_data():
    """Load models and validation data"""
    models = load_models()
    # Adjust the path to your validation data as needed
    data_dir = Path(__file__).resolve().parent.parent
    try:
        val_df = pd.read_csv("stroke_data_smoted_scaled_for_pycaret.csv")
        # Assume the target column is named 'stroke'
        X_val = val_df.drop(columns=['stroke'])
        y_val = val_df['stroke']
    except Exception as e:
        st.error(f"Validation data loading failed: {str(e)}")
        st.stop()
    return models, X_val, y_val

# --- Main Page ---
def model_performance_page():
    st.set_page_config(layout="wide")
    st.title("🧪 Live Model Evaluation")
    
    # Load models and validation data
    models, X_val, y_val = load_models_and_data()
    
    # Calculate all metrics
    with st.spinner("Computing live metrics..."):
        models_metrics = {}
        for name, model in models.items():
            metrics = evaluate_model(model, X_val, y_val)
            if metrics is not None:
                models_metrics[name] = metrics
    
    if not models_metrics:
        st.error("No models could be evaluated")
        return
    
    # --- Performance Dashboard ---
    with st.expander("📊 Performance Dashboard", expanded=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Key Metrics")
            metric_choice = st.selectbox(
                "Select metric", 
                ['accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']
            )
            
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(6, 4))
            names = list(models_metrics.keys())
            values = [m.get(metric_choice, 0) for m in models_metrics.values()]
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bars = ax.bar(names, values, color=colors[:len(names)])
            ax.set_title(f'{metric_choice.upper()} Comparison')
            ax.set_ylim(0, 1)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("### ROC Curves")
            st.pyplot(plot_roc_curves(models, X_val, y_val))
    
    # --- Detailed Analysis ---
    with st.expander("🔍 Model-Specific Analysis"):
        model_choice = st.selectbox("Select model", list(models_metrics.keys()))
        metrics = models_metrics[model_choice]
        
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Low Risk', 'High Risk'])
        ax.set_yticklabels(['Low Risk', 'High Risk'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        st.pyplot(fig)
        
        # Classification Report
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'))
    
    # --- Feature Importance ---
    with st.expander("📌 Feature Importance"):
        model_choice = st.selectbox(
            "Select model for feature importance", 
            [name for name, model in models.items() 
                if hasattr(model, 'feature_importances_')],
            key="feature_importance_selector"
        )
        
        if model_choice in models:
            model = models[model_choice]
            importance = pd.DataFrame({
                'Feature': X_val.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            importance.head(10).plot.barh(x='Feature', y='Importance', ax=ax, color='#1f77b4')
            ax.set_title(f'Top 10 Features - {model_choice}')
            st.pyplot(fig)
            
            st.dataframe(importance.head(20).style.background_gradient(cmap='Blues'))

if __name__ == "__main__":
    model_performance_page()