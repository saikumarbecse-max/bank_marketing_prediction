import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score,precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef
)
from sklearn.metrics import classification_report

import warnings 
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Bank Marketing Classifier", layout="centered")

# App title and description
st.title("📊 Bank Marketing Classification App")
st.write("Upload **TEST DATA ONLY** (CSV format)")

# File uploader for test CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

#
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

#show only if model choice is Logistic Regression
if model_choice == "Logistic Regression":
    threshold_choice = st.selectbox(
        "Select Probability Threshold",
    [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], index=2
    )

def calc_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    return {
        "Confusion Matrix": cm,
        "Accuracy": accuracy,
        "AUC-ROC": auc_roc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }

def display_metrics(metrics,y_test,y_pred):

    left_col, right_col = st.columns(2)

    #---------- METRICS DISPLAY ---------
    with left_col:
        st.subheader("📈 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['Precision']:.3f}")
        col3.metric("Recall", f"{metrics['Recall']:.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        col5.metric("AUC-ROC", f"{metrics['AUC-ROC']:.3f}")
        col6.metric("MCC", f"{metrics['MCC']:.3f}")

        st.markdown("---")

    # ---------- CONFUSION MATRIX (PLOT) ---------
    with right_col:
        st.subheader("🔍 Confusion Matrix")
        cm = metrics["Confusion Matrix"]
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        st.markdown("---")

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file, sep=',')

        # Preprocess the test data
        # Assuming the same preprocessing steps as training
        X_test = test_data.drop(columns=['y'])
        y_test = test_data['y']

        # Make predictions based on selected model call this method predict_logistic_regression from logistic_model.py
        
        if model_choice == "Logistic Regression":
            model = joblib.load('pkl/logistic_model.pkl')
            y_predproba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_predproba >= threshold_choice).astype(int)
            metrics = calc_metrics(y_test,y_pred)
            
        elif model_choice == "Decision Tree":
            model = joblib.load('pkl/decision_tree_model.pkl')
            y_pred = model.predict(X_test)
            metrics = calc_metrics(y_test,y_pred)

        if metrics:
            display_metrics(metrics,y_test,y_pred)


    except Exception as e:
        st.error(f"Error processing file: {e}")