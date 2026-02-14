import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef, classification_report
)

import warnings 
warnings.filterwarnings('ignore')

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Bank Marketing Prediction", 
    page_icon="🏦",
    layout="wide"
)

# ========================================
# TITLE AND DESCRIPTION
# ========================================
st.title("🏦 Bank Marketing Campaign Prediction")
st.markdown("### Predict Term Deposit Subscription Using Machine Learning")
st.markdown("---")

# ========================================
# SIDEBAR - MODEL SELECTION
# ========================================
st.sidebar.header("⚙️ Model Configuration")

# Model selection dropdown
model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Model file mapping
MODEL_FILES = {
    "Logistic Regression": "models/logistic_model.pkl",
    "Decision Tree": "models/decision_tree_model.pkl",
    "K-Nearest Neighbors": "models/knn_model.pkl",
    "Naive Bayes": "models/nb_model.pkl",
    "Random Forest": "models/rf_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}

# Load selected model
@st.cache_resource
def load_model(model_path):
    """Load a saved model from disk"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_FILES[model_choice])

if model is not None:
    st.sidebar.success(f"✅ {model_choice} loaded successfully!")
else:
    st.sidebar.error("❌ Failed to load model")
    st.stop()

# Threshold selection (optional - only for demonstration)
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Prediction Threshold")
st.sidebar.caption("Default: 0.5 (recommended)")

use_custom_threshold = st.sidebar.checkbox("Use custom threshold", value=False)
if use_custom_threshold:
    threshold = st.sidebar.slider(
        "Select threshold",
        min_value=0.3,
        max_value=0.7,
        value=0.5,
        step=0.05
    )
else:
    threshold = 0.5

st.sidebar.markdown("---")

# ========================================
# INFORMATION SECTION
# ========================================
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### Bank Marketing Prediction Application
    
    This application predicts whether a customer will subscribe to a term deposit 
    based on the Bank Marketing dataset from UCI Machine Learning Repository.
    
    **How to use:**
    1. Select a machine learning model from the sidebar
    2. Upload your test data (CSV format)
    3. Click "Generate Predictions" to see results
    
    **Expected CSV format:**
    - Must contain all preprocessed features (same as training data)
    - Optional: Include 'y' column (0/1) for model evaluation
    - Features should be already encoded and scaled
    
    **Models Available:**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors (kNN)
    - Naive Bayes
    - Random Forest
    - XGBoost
    
    **Evaluation Metrics:**
    - Accuracy, Precision, Recall, F1 Score
    - AUC-ROC, MCC (Matthews Correlation Coefficient)
    - Confusion Matrix
    - Classification Report
    """)

# ========================================
# METRICS CALCULATION FUNCTION
# ========================================
def calc_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive evaluation metrics
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_pred_proba: Predicted probabilities for AUC calculation
    
    Returns:
    - Dictionary with all metrics
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate AUC-ROC using probabilities
    if y_pred_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc_roc = None
    else:
        auc_roc = None
    
    return {
        "Confusion Matrix": cm,
        "Accuracy": accuracy,
        "AUC-ROC": auc_roc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }

# ========================================
# DISPLAY METRICS FUNCTION
# ========================================
def display_metrics(metrics, y_test, y_pred):
    """Display evaluation metrics in a organized layout"""
    
    st.header("📊 Model Evaluation Results")
    st.markdown("---")
    
    # Performance Metrics in columns
    st.subheader("📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    
    with col2:
        st.metric("Recall", f"{metrics['Recall']:.4f}")
        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    
    with col3:
        if metrics['AUC-ROC'] is not None:
            st.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
        else:
            st.metric("AUC-ROC", "N/A")
        st.metric("MCC", f"{metrics['MCC']:.4f}")
    
    st.markdown("---")
    
    # Create two columns for confusion matrix and classification report
    col_left, col_right = st.columns(2)
    
    # Confusion Matrix
    with col_left:
        st.subheader("🔥 Confusion Matrix")
        cm = metrics["Confusion Matrix"]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"Confusion Matrix - {model_choice}", fontsize=12)
        
        st.pyplot(fig)
        plt.close()
        
        # Explanation
        TN, FP, FN, TP = cm.ravel()
        st.caption(f"""
        **Matrix Interpretation:**
        - True Negatives (TN): {TN}
        - False Positives (FP): {FP}
        - False Negatives (FN): {FN}
        - True Positives (TP): {TP}
        """)
    
    # Classification Report
    with col_right:
        st.subheader("📋 Classification Report")
        
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=['No (0)', 'Yes (1)'],
            output_dict=True,
            zero_division=0
        )
        
        report_df = pd.DataFrame(report).transpose()
        
        # Style the dataframe
        styled_df = report_df.style.background_gradient(
            cmap='RdYlGn', 
            subset=['precision', 'recall', 'f1-score']
        ).format(precision=3)
        
        st.dataframe(styled_df, use_container_width=True)
        
        st.caption("""
        **Metrics Explanation:**
        - **Precision**: Of all predicted positives, how many are correct?
        - **Recall**: Of all actual positives, how many did we catch?
        - **F1-Score**: Harmonic mean of precision and recall
        - **Support**: Number of samples in each class
        """)

# ========================================
# FILE UPLOAD SECTION
# ========================================
st.header("📁 Upload Test Data")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload preprocessed test data with the same features as training data"
    )

with col2:
    st.markdown("**File Requirements:**")
    st.caption("✅ CSV format")
    st.caption("✅ Preprocessed features")
    st.caption("✅ Optional: 'y' column for evaluation")

# ========================================
# SAMPLE DATA SECTION
# ========================================
with st.expander("📖 Expected Data Format"):
    st.markdown("""
    ### Test Data Format
    
    Your CSV file should contain:
    - All preprocessed features (one-hot encoded categorical variables)
    - Scaled numerical features
    - Optional: 'y' column with values 0 or 1
    
    **Example columns:**
    ```
    age, duration, campaign, pdays, previous, emp.var.rate, 
    cons.price.idx, cons.conf.idx, euribor3m, nr.employed,
    job_admin., job_blue-collar, ..., y
    ```
    
    **Note:** The features must match exactly with the training data format.
    """)

# ========================================
# PREDICTION AND EVALUATION
# ========================================
if uploaded_file is not None:
    try:
        # Read uploaded file
        test_data = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
        
        # Show data preview
        with st.expander("👀 View Data Preview (First 5 rows)"):
            st.dataframe(test_data.head())
        
        # Check if target column exists
        has_labels = 'y' in test_data.columns
        
        if has_labels:
            st.info("ℹ️ Target column 'y' found - Model will be evaluated")
            X_test = test_data.drop(columns=['y'])
            y_test = test_data['y']
        else:
            st.warning("⚠️ No target column 'y' found - Only predictions will be generated")
            X_test = test_data
            y_test = None
        
        # Generate predictions button
        if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
            
            with st.spinner(f"Making predictions using {model_choice}..."):
                
                try:
                    # Get predictions and probabilities
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        y_pred = (y_pred_proba >= threshold).astype(int)
                    else:
                        y_pred = model.predict(X_test)
                        y_pred_proba = None
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display prediction summary
                    st.header("📊 Prediction Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Samples", len(y_pred))
                    with col2:
                        predicted_yes = int(sum(y_pred))
                        st.metric("Predicted 'Yes' (Subscribe)", predicted_yes)
                    with col3:
                        predicted_no = len(y_pred) - predicted_yes
                        st.metric("Predicted 'No'", predicted_no)
                    
                    # Prediction distribution chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    pred_counts = pd.Series(y_pred).value_counts().sort_index()
                    pred_counts.index = ['No (0)', 'Yes (1)']
                    
                    colors = ['#ff6b6b', '#51cf66']
                    pred_counts.plot(kind='bar', ax=ax, color=colors)
                    ax.set_title('Prediction Distribution', fontsize=14)
                    ax.set_ylabel('Count', fontsize=11)
                    ax.set_xlabel('Prediction', fontsize=11)
                    plt.xticks(rotation=0)
                    
                    for i, v in enumerate(pred_counts):
                        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=10)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("---")
                    
                    # Show predictions table
                    with st.expander("📋 View All Predictions"):
                        results_df = X_test.copy()
                        results_df['Predicted'] = y_pred
                        results_df['Predicted_Label'] = results_df['Predicted'].map({0: 'No', 1: 'Yes'})
                        
                        if y_pred_proba is not None:
                            results_df['Probability'] = y_pred_proba
                        
                        if has_labels:
                            results_df['Actual'] = y_test.values
                            results_df['Actual_Label'] = results_df['Actual'].map({0: 'No', 1: 'Yes'})
                            results_df['Correct'] = (results_df['Predicted'] == results_df['Actual'])
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download predictions
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f'predictions_{model_choice.replace(" ", "_").lower()}.csv',
                            mime='text/csv'
                        )
                    
                    # If labels exist, calculate and display metrics
                    if has_labels and y_test is not None:
                        st.markdown("---")
                        metrics = calc_metrics(y_test, y_pred, y_pred_proba)
                        display_metrics(metrics, y_test, y_pred)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.exception(e)
                    st.info("💡 Make sure your test data has the same features as training data")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to begin predictions")

# ========================================
# SIDEBAR - ADDITIONAL INFO
# ========================================
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Bank Marketing Prediction**

This application predicts whether a client will subscribe to a term deposit 
based on marketing campaign data.

**Models Available:**
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest
- XGBoost

**Dataset:** Bank Marketing Dataset (UCI ML Repository)

**Course:** M.Tech (AIML/DSE)  
**Assignment:** ML Assignment 2
""")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Use threshold 0.5 for fair comparison across models")
st.sidebar.caption("📊 All 6 metrics calculated: Accuracy, AUC, Precision, Recall, F1, MCC")