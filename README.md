# 🚀 Stroke Prediction Model - Machine Learning & Deep Learning

## 📌 Overview
This project builds a **stroke prediction model** using multiple machine learning and deep learning techniques. It includes **data preprocessing, feature engineering, hyperparameter tuning, model calibration, and explainability techniques** to enhance predictive performance.

## 📊 Dataset
- **Source:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
- **Features:** 20 columns, including patient demographics and health records.  
- **Objective:** Predict the likelihood of a **stroke (binary classification problem).**  

## 🏗️ Key Features
✅ **Feature Engineering & Preprocessing:** One-hot encoding, missing value imputation, outlier handling, and feature selection.  
✅ **Data Balancing:** SMOTE, Random Undersampling, and SMOTETomek to handle class imbalance.  
✅ **Models Implemented:**  
   - 🧠 **Neural Network** (TensorFlow/Keras)  
   - 🚀 **XGBoost**  
   - 🌲 **Random Forest**  
   - 📊 **Logistic Regression**  
✅ **Hyperparameter Tuning:** Using **Optuna** for optimizing model parameters.  
✅ **Model Calibration:** **Isotonic Regression** for improving probability estimates.  
✅ **Explainability:** **SHAP** (Shapley Additive Explanations) for feature importance analysis.  
✅ **Performance Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Brier Score.  

## 📌 Model Performance
| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision |
|----------------------|----------|------------|---------|----------|---------|--------------|
| 🚀 **XGBoost**            | **0.971992** | 0.986111   | 0.957469 | 0.971579 | **0.995290** | **0.996051** |
| 🌲 **Random Forest**      | **0.974585** | **0.990354**   | 0.958506 | **0.974170** | 0.993180 | 0.990124 |
| 📊 **Logistic Regression**| 0.808091 | 0.781784   | 0.854772 | 0.816650 | 0.873493 | 0.836539 |
| 🧠 **Neural Network (NN)** | 0.950207 | 0.934870   | **0.967842** | 0.951070 | 0.981484 | 0.974995 |

## 📌 Detailed Classification Report
### 🧠 Neural Network:
```
              precision    recall  f1-score   support

           0       0.97      0.93      0.95       964
           1       0.93      0.97      0.95       964

    accuracy                           0.95      1928
   macro avg       0.95      0.95      0.95      1928
weighted avg       0.95      0.95      0.95      1928
```
### 🚀 XGBoost:
```
              precision    recall  f1-score   support

           0       0.96      0.99      0.97       964
           1       0.99      0.96      0.97       964

    accuracy                           0.97      1928
   macro avg       0.97      0.97      0.97      1928
weighted avg       0.97      0.97      0.97      1928
```
### 🌲 Random Forest:
```
              precision    recall  f1-score   support

           0       0.96      0.99      0.97       964
           1       0.99      0.96      0.97       964

    accuracy                           0.97      1928
   macro avg       0.98      0.97      0.97      1928
weighted avg       0.98      0.97      0.97      1928
```
### 📊 Logistic Regression:
```
              precision    recall  f1-score   support

           0       0.84      0.76      0.80       964
           1       0.78      0.85      0.82       964

    accuracy                           0.81      1928
   macro avg       0.81      0.81      0.81      1928
weighted avg       0.81      0.81      0.81      1928
```

## 📢 Results & Insights
- 🚀 **Random Forest achieved the highest accuracy (97.46%)**, while **XGBoost performed best in overall AUC (99.52%)**.
- 📊 **Feature importance analysis** highlighted `age`, `bmi`, `heart_disease`, `avg_glucose_level`, and `hypertension` as the most impactful features.
- 🎯 **Model calibration improved probability predictions**, reducing **Brier Score** and **Expected Calibration Error (ECE)**.
- 🔍 **SHAP analysis** provided insights into how different features influence stroke predictions.