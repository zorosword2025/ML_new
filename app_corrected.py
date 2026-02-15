"""
ML Assignment 2 - Streamlit App (Corrected Version)
Properly loads test_data.csv or test.csv
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
# NEW: Download Test Data Option
# ============================================================================
st.header("0. Download Test Data (Optional)")

# Try to load test data from different possible filenames
test_df = None
test_file = None

# Check for different possible filenames
possible_files = ['test_data.csv', 'test.csv', 'test_dataset.csv']

for filename in possible_files:
    if os.path.exists(filename):
        try:
            test_df = pd.read_csv(filename)  # ← THIS IS HOW YOU LOAD THE CSV
            test_file = filename
            break
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")

# If test data was found and loaded
if test_df is not None:
    st.success(f"✅ Found and loaded: {test_file}")
    
    # Show basic info
    st.write(f"Available test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Rows", f"{test_df.shape[0]:,}")
    with col2:
        st.metric("📋 Total Columns", f"{test_df.shape[1]}")
    with col3:
        if 'target' in test_df.columns:
            target_counts = test_df['target'].value_counts()
            st.metric("🎯 Target Classes", f"0:{target_counts.get(0,0)} | 1:{target_counts.get(1,0)}")
    
    # Download button
    csv = test_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download test_data.csv",
        data=csv,
        file_name="test_data.csv",
        mime="text/csv"
    )
    
    # Show sample
    if st.checkbox("Show sample data"):
        st.dataframe(test_df.head())
else:
    st.warning("⚠️ No test data file found!")
    st.info("Checked for: test_data.csv, test.csv, test_dataset.csv")
    st.markdown("""
    **To fix this:**
    1. Run `python train_models.py` to generate test_data.csv
    2. Or run `python generate_test_data.py` for sample data
    3. Or upload your own CSV file below
    """)

st.markdown("---")

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
        
        # Check if model file exists
        if not os.path.exists(MODEL_FILES[selected_model]):
            st.error(f"❌ Model file not found: {MODEL_FILES[selected_model]}")
            st.info("Please train the models first by running train_models.py")
        else:
            # Load selected model
            model = load_model(MODEL_FILES[selected_model])
            
            try:
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
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.code(str(e))
    
    else:
        st.error("Error: The uploaded CSV must contain a 'target' column.")

else:
    st.info("Please upload a CSV file to begin.")
