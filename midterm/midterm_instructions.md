## Canvas Instructions

**Our primary goal is to minimize fraudulent credit card transactions.**

For the Midterm Exam, use the roadmap for building machine learning systems and the sklearn library. You are required to use the original unbalanced dataset (creditcard_small.csv) and the balanced dataset (which you will create). I have reduced the dataset from approximately 300,000 samples to 100,000 samples for efficiency (so you do not have to wait a long time during computation, no step should take more than a few minutes on https://coding.csel.ioLinks to an external site.) . For the same reason, use k = 5 for CV and do not perform pairwise plot. There are 28 anonymized features (V1 - V28). We don't need to understand the features. That is the beauty of ML! It will discern the pattern regardless of what the features mean. Class is a binary label indicating whether the transaction is genuine (0) or fraudulent (1). The features are already scaled, so you do not need to scale them. You are required to  train the 3 algorithms (DT, kNN, SVM) that we have studied in this class. So, in essence you will be performing 6 training runs (2 datasets X 3 algorithms) 

Perform steps 1- 3 of EDA:

1. Performing preliminary checks, creating balanced dataset, identifying and rectifying outliers, missing values and statistics, and creating correlation matrices. [10]

Perform steps 1- 6 of training and testing of 3 algorithms with the 2 datasets:

2. Unbalanced dataset and DT. [10]

3. Unbalanced dataset and kNN. [10]

4. Unbalanced dataset and SVM. [10]

5. Balanced dataset and DT. [10]

6. Balanced dataset and kNN. [10]

7. Balanced dataset and SVM. [10]

Each of these training testing runs must produce a report similar to:

```bash
Classification Report: KNN 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19959
           1       0.86      0.78      0.82        41

    accuracy                           1.00     20000
   macro avg       0.93      0.89      0.91     20000
weighted avg       1.00      1.00      1.00     20000

Training accuracy score of KNN is 99.95
Roc_Auc training score for KNN is 91.2: 
Test accuracy score of KNN is 99.87
Roc_Auc test score for KNN is 78.03: 

[[19951     8]
 [   18    23]]

Accuracy: 0.999
Precision: 0.865
Recall: 0.780
F1: 0.821

k-fold crossvalidation Average:  0.99946999699985
```

8. Create Learning Curves for DT, kNN and SVM [5]

9. Perform a Grid Search (with C and gamma = [0.01, 0.1, 1.0, 10.0] and kernel = ['rbf', 'linear']) and produce best score and parameters (optimal parameters for SVC) for Balanced Dataset [5]

10. Create ROC/AUC curve for the optimal SVC [5]

**Analysis, Observations and Conclusions:**

11. Analyze the correlation: Which top 4 features coorelate to Genuine transactions and which top 4 features coorrelate to Fraudulent transactions? [2.5]

12. Analyze Classification Reports for Unbalanced Dataset: What are you observations? [2.5]

13. Analyze Classification Reports for Balanced Dataset: What are your observations? [2.5]

14. Analyze Classification Reports from Unbalanced Dataset and Balanced Dataset: What are your observations? [2.5]

15. Analyze Learning curves: What do these curves tell us about bias and variance? [2.5]

16. Your overall conclusions and recommendations for meeting the primary goal. [2.5]

**NOTE:** An observation is a meaningful insight. Stating that Sensitivity = TP/FN+TP, although correct, is NOT a meaningful insight [0 points]. Here is an example of a meaningful insight: Our primary goal is to minimize the spread of a communicable disease (COVID-19), thus high sensitivity is desired for tests, and SVC is a better model than DT because it minimizes FN.   

After you have completed Midterm, upload and submit your files.

For notebook assignments, you will always turn in your .ipynb notebook file and an HTML file of the notebook.

For HTML: Click File → Save and Export Notebook As → HTML to make a nice HTML of the notebook.

For .ipynb: Click File → Download

Turn in both the .ipynb and .html files in Canvas.