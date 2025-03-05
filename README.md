# 🚀 Stroke Prediction Model - Machine Learning & Deep Learning

## 📌 Overview
This project builds a **stroke prediction model** using multiple machine learning and deep learning techniques. It includes data preprocessing, feature engineering, hyperparameter tuning, model calibration, and explainability techniques.

## 📊 Dataset
- Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Features: 22 columns, including patient demographics and health records.
- Objective: Predict the likelihood of a **stroke (binary classification problem)**.

## 🏗️ Key Features
✅ **Feature Engineering & Preprocessing**: One-hot encoding, missing value imputation, outlier handling, and feature selection.  
✅ **Data Balancing**: SMOTE, Random Undersampling, and SMOTETomek to handle class imbalance.  
✅ **Models Implemented**:  
   - **Neural Network** (TensorFlow/Keras)  
   - **XGBoost**  
   - **Random Forest**  
   - **Logistic Regression**  
✅ **Hyperparameter Tuning**: Using **Optuna** for optimizing model parameters.  
✅ **Model Calibration**: **Isotonic Regression** for improving probability estimates.  
✅ **Explainability**: **SHAP** (Shapley Additive Explanations) for feature importance analysis.  
✅ **Performance Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC, Brier Score.  

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/stroke-prediction.git
cd stroke-prediction

# Install required dependencies
pip install -r requirements.txt
