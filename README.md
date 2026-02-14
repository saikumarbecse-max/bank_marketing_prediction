# bank_marketing_prediction
Bank Marketing Prediction

a. Problem statement 
  The objective of this project is to predict whether a client will subscribe to a term deposit based on direct marketing campaign data from a Portuguese banking institution. This is a binary classification problem where the goal is to identify potential customers who are likely to subscribe to a term deposit product. 
  The prediction model helps the bank:
    Optimize marketing campaign effectiveness
    Reduce marketing costs by targeting likely subscribers
    Improve customer satisfaction through personalized outreach
    Allocate resources more efficiently
    
b. Dataset description 
    Dataset Name: Bank Marketing Dataset
    Source: UCI Machine Learning Repository
    URL: https://archive.ics.uci.edu/ml/datasets/bank+marketing
    Dataset Characteristics
      Total Samples: 41,188 customer records
      Number of Features: 20 (before encoding)
      Features after Preprocessing: 62 (after one-hot encoding)
      Target Variable: y (binary: yes/no for term deposit subscription)
      Problem Type: Binary Classification
| Class                   | Count  | Percentage |
|-------------------------|--------|------------|
| No subscription (0)     | 36,548 | 88.7%      |
| Subscribed (1)          | 4,640  | 11.3%      |
    Imbalance Ratio7.88:1    

c. Models used:
  Model Performance Comparison
| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8654 | 0.9438 | 0.4517 | 0.9116 | 0.6041 | 0.5817 |
| Decision Tree | 0.8438 | 0.9401 | 0.4123 | 0.9095 | 0.5674 | 0.5450 |
| K-Nearest Neighbors | 0.9104 | 0.9078 | 0.6594 | 0.4235 | 0.5157 | 0.4829 |
| Naive Bayes | 0.8301 | 0.8406 | 0.3604 | 0.6563 | 0.4652 | 0.3980 |
| Random Forest | 0.8407 | 0.9448 | 0.4110 | 0.9569 | 0.5751 | 0.5280 |
| XGBoost | 0.8708 | 0.9536 | 0.4636 | 0.9343 | 0.6197 | 0.6016 |

  Model Observations Table:
  ## 🔎 Model Observations

| ML Model | Observation about Model Performance |
|---------|--------------------------------------|
| **Logistic Regression** | Achieves strong balanced performance with **86.5% accuracy** and excellent **AUC of 0.944**. The use of `class_weight='balanced'` effectively handles the **7.88:1 class imbalance**. Very high **recall (91.2%)** makes it suitable where missing positive cases is costly. The `solver='liblinear'` is well-suited for binary classification. Provides excellent interpretability through feature coefficients. **F1 score of 0.60** shows a reasonable precision–recall balance. |
| **Decision Tree** | Configured with `max_depth=5` and `min_samples_leaf=100` to prevent overfitting on imbalanced data. Achieves **84.4% accuracy** with outstanding **AUC of 0.940**. Very high **recall (90.9%)** but lower **precision (41.2%)** indicates conservative minority predictions. `class_weight='balanced'` ensures minority class weighting. Shallow depth improves generalization. Despite being a single tree, achieves AUC comparable to ensemble methods. |
| **kNN** | Achieves highest **accuracy (91.0%)** using distance-weighted voting with `n_neighbors=10`. Shows precision–recall trade-off with high **precision (65.9%)** but lowest **recall (42.4%)**. Highly selective positive predictions → fewer false positives but more missed subscriptions. Minkowski distance (`p=2`, Euclidean) works well with standardized features. Strongly dependent on feature scaling. Lazy learner → slower prediction on large datasets but no training time. |
| **Naive Bayes** | Shows lowest overall performance with **83.0% accuracy** and **MCC of 0.398**. Feature independence assumption likely violated (correlated economic indicators). Despite limitations, achieves reasonable **recall (65.6%)**. **AUC of 0.841** is lowest, indicating moderate ranking ability. Extremely fast training/prediction → useful baseline. Gaussian assumption may not perfectly match post–one-hot encoded features. |
| **Random Forest (Ensemble)** | Ensemble of **300 trees** achieves highest **recall (95.7%)** and excellent **AUC of 0.945**. Captures nearly all potential subscribers → valuable for campaign reach. Configured with `max_depth=15`, `min_samples_leaf=50`. `class_weight='balanced'` handles imbalance. Lower **precision (41.1%)**, but high recall makes it ideal for initial screening. Bagging reduces variance vs single Decision Tree. |
| **XGBoost (Ensemble)** | **Best overall performer** with highest **MCC (0.602)**, **F1 Score (0.620)**, and **AUC (0.954)**. `scale_pos_weight=7.88` effectively handles imbalance. Conservative `learning_rate=0.05` with **300 estimators** ensures stable optimization. Excellent balance: **precision (46.4%)** & **recall (93.4%)** → reliable for deployment. Sequential error correction captures minority patterns. `eval_metric='logloss'` improves probability calibration. |
  
