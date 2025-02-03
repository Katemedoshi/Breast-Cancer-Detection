# Breast-Cancer-Detection
Project Overview
This project is a machine learning-based classification model designed to predict whether a breast cancer tumor is malignant or benign. The model is trained on a dataset containing various tumor characteristics and uses classification algorithms to make predictions.

Dataset:The dataset used in this project contains features related to tumor measurements.It includes attributes such as mean radius, texture, perimeter, area, smoothness, etc.The target variable classifies tumors as malignant (1) or benign (0).

Libraries Used:The following Python libraries are used in this notebook:

1.pandas – for data manipulation
2.numpy – for numerical operations
3.matplotlib & seaborn – for data visualization
4.sklearn (Scikit-learn) – for model training, evaluation, and preprocessing

Steps in the Notebook
1.Data Loading
2.The dataset is imported into a pandas DataFrame.
3.Exploratory Data Analysis (EDA)
4.Data visualization and statistical insights are extracted.
5.Correlation analysis is performed to identify important features.
6.Data Preprocessing
7.Handling missing values (if any).
8.Encoding categorical variables (if applicable).
9.Feature scaling using StandardScaler or MinMaxScaler.
10.Model Selection & Training
11.Different classification models are tested (e.g., Logistic Regression, Decision Tree, Random Forest, SVM, etc.).
12.The best-performing model is selected based on evaluation metrics.

Model Evaluation
Performance is assessed using accuracy, precision, recall, F1-score, and confusion matrix.
Predictions & Conclusion
The trained model is used to make predictions on test data.

