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
    Class Distribution
      Class                  Count        Percentage
      No subscription (0)    36,548          88.7%
      Subscribed (1)         4,640           11.3%
    Imbalance Ratio7.88:1    

c. Models used:

  Make a Comparison Table with the evaluation metrics calculated for all the 6  models as below:  
  ML Model              Accuracy        AUC      Precision  Recall    F1 Score    MCC
  Logistic Regression  0.8654          0.9438      0.4517    0.9116    0.6041     0.5817
  Decision Tree        0.8438          0.9401      0.4123    0.9095    0.5674     0.5450
  K-Nearest Neighbors  0.9104          0.9078      0.6594    0.4235    0.5157     0.4829
  Naive Bayes          0.8301          0.8406      0.3604    0.6563    0.4652     0.3980
  Random Forest        0.8407          0.9448      0.4111    0.9569    0.5751     0.5628
  XGBoost              0.8708          0.9536      0.4636    0.9343    0.6197     0.6016

  Model Observations Table:
  ML Model Name             Observation about model performance 
Logistic Regression       Achieves strong balanced performance with 86.5% accuracy and excellent AUC of 0.944. The use of class_weight='balanced' effectively handles the 7.88:1 class                                imbalance. Very high recall (91.2%) makes it suitable for applications where missing positive cases is costly. The solver='liblinear' is well-suited for this                               binary classification problem. Provides excellent interpretability through feature coefficients, allowing stakeholders to understand which factors drive                                    subscription likelihood. F1 score of 0.60 demonstrates reasonable balance between precision and recall. 

Decision Tree             Configured with max_depth=5 and min_samples_leaf=100 to prevent overfitting on imbalanced data. Achieves 84.4% accuracy with outstanding AUC of 0.940. Very high                            recall (90.9%) but lower precision (41.2%) indicates the model is conservative in predicting the minority class. The class_weight='balanced' parameter ensures                              appropriate weight is given to minority class samples. The shallow tree depth successfully trades some training accuracy for better generalization capability.                              Despite being a single tree, achieves AUC comparable to ensemble methods. 

kNN                       Achieves highest accuracy (91.0%) among all models using distance-weighted voting with n_neighbors=10. Shows distinct precision-recall trade-off with high                                  precision (65.9%) but lowest recall (42.4%). The model is highly selective in predicting positive cases, resulting in fewer false positives but missing more                                actual subscriptions. Minkwski distance metric (p=2, Euclidean) works effectively with standardized features. Performance is heavily dependent on feature scaling                           quality. Being a lazy learner, all computation happens at prediction time, making it slower for large datasets but eliminating training time. 

Naive Bayes               Shows lowest overall performance with 83.0% accuracy and MCC of 0.398. The assumption of feature independence likely doesn't hold well for this dataset,                                    particularly among correlated economic indicators (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed). Despite limitations, achieves reasonable                           recall (65.6%). AUC of 0.841 is the lowest, indicating moderate ranking capability. Provides extremely fast training and prediction times, making it valuable as                            a quick baseline. The Gaussian assumption may not perfectly match the actual distribution of features after one-hot encoding. 

Random Forest (Ensemble) Ensemble of 300 trees achieves highest recall (95.7%) among all models and excellent AUC of 0.945. Catches nearly all potential subscribers, critical for                                   maximizing marketing campaign reach. Configuration with max_depth=15 and min_samples_leaf=50 allows deeper trees while controlling overfitting through ensemble                             averaging. The class_weight='balanced' ensures each tree appropriately weights minority class samples. Despite lower precision (41.1%), high recall makes this                              excellent for initial customer screening. Bootstrap aggregation reduces variance compared to single Decision Tree, providing more robust predictions. 

XGBoost (Ensemble)       Best overall performer with highest MCC (0.602), F1 Score (0.620), and AUC (0.954). Gradient boosting with scale_pos_weight=7.88 provides most effective handling                           of class imbalance. Conservative learning_rate=0.05 with 300 estimators ensures stable optimization without overfitting. Achieves excellent balance between                                 precision (46.4%) and recall (93.4%), making it most reliable for practical deployment. Sequential error correction is particularly effective at identifying                                minority class patterns. The eval_metric='logloss' optimizes for probabilistic predictions, resulting in well-calibrated probability estimates and highest AUC                              score.

  
