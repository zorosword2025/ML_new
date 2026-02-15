"""
ML Assignment 2 - Streamlit App (Minimal Version)
Contains ONLY the 4 required features + Download option
"""

import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("Adult Income Prediction - ML Models")

# Model files dictionary
MODEL_FILES = {
    'Logistic Regression': 'model_logistic_regression.pkl',
    'Decision Tree': 'model_decision_tree.pkl',
    'K-Nearest Neighbor': 'model_k_nearest_neighbor.pkl',
    'Naive Bayes': 'model_naive_bayes.pkl',
    'Random Forest': 'model_random_forest.pkl',
    'XGBoost': 'model_xgboost.pkl'
}

# Load model function
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)



# ============================================================================
# FEATURE 1: Dataset Upload Option (CSV) - 1 mark
# ============================================================================
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.success(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Check if target column exists
    if 'target' in data.columns:
        # Separate features and target
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        # ============================================================================
        # FEATURE 2: Model Selection Dropdown - 1 mark
        # ============================================================================
        st.header("2. Select Model")
        selected_model = st.selectbox("Choose a model:", list(MODEL_FILES.keys()))
        
        # Load selected model
        model = load_model(MODEL_FILES[selected_model])
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # ============================================================================
        # FEATURE 3: Display of Evaluation Metrics - 1 mark
        # ============================================================================
        st.header("3. Evaluation Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
        
        with col2:
            st.metric("AUC Score", f"{auc:.4f}")
            st.metric("Recall", f"{recall:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC Score", f"{mcc:.4f}")
        
        # ============================================================================
        # FEATURE 4: Confusion Matrix or Classification Report - 1 mark
        # ============================================================================
        st.header("4. Confusion Matrix & Classification Report")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Confusion Matrix
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['<=50K', '>50K'],
                       yticklabels=['<=50K', '>50K'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
        
        # Classification Report
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred,
                                          target_names=['<=50K', '>50K'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))
    
    else:
        st.error("Error: The uploaded CSV must contain a 'target' column.")

else:
    st.info("Please upload a CSV file to begin.")
