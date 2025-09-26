# This is the Python code to implement the model training with Gridsearch and K-fold cross-validation, and, compute Out-of-Fold SHAP values  per sample

from google.colab import drive
drive.mount('/content/drive')

# 2. Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import shap


# print(shap.__version__)

# 3. Load and Preprocess Data

# Load dataset
data_cleaned = pd.read_csv('/content/data_cleaned_w_cluster.csv')

# Remove first column, in case contains the index from the ouput from R
# data_cleaned.drop(columns=data_cleaned.columns[:2], axis=1, inplace=True)

# Use this code to create 4 or 6 groups of risk based on BMI
# Define BMI categories
# def categorize_bmi_risk (bmi):
#     if bmi < 25 :
#         return '< 25'
#     elif 25 <= bmi < 30:
#         return '25-30'
#     else:
#         return '> 30'

# data_cleaned['BMI_Classification'] = data_cleaned['BMI'].apply(categorize_bmi_risk)


# Perform one-hot encoding for vital_status or DSS
# data_cleaned['vital_status_encoded'] = data_cleaned['vital_status'].map({'Alive': 0, 'Dead': 1})
data_cleaned['DSS_status'] = data_cleaned['DSS'].map({0 : 'DSS_Alive', 1: 'DSS_Dead'})

# np.unique(y)

# Handle missing values
data_cleaned.fillna(data_cleaned.median(numeric_only=True), inplace=True)  # Impute numerical values

*2*. Define model & hyperparameter grid

param_grid = {
    'num_leaves': [15, 31],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

lgb_estimator = LGBMClassifier(random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=lgb_estimator,
                           param_grid=param_grid,
                           cv=cv_inner,
                           scoring='roc_auc',
                           verbose=0,
                           n_jobs=-1)


3. Outer cross-validation loop with SHAP value computatio

cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

shap_values_all_folds = []
y_true_all_folds = []
y_pred_all_folds = []

for train_idx, test_idx in cv_outer.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Grid search to find best model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict and store results
    y_pred = best_model.predict_proba(X_test)[:, 1]
    y_pred_all_folds.extend(y_pred)
    y_true_all_folds.extend(y_test)

    # SHAP values (TreeExplainer)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)[1]  # Class 1
    shap_values_all_folds.append(shap_values)



# Evaluate the XGBoost model's performance
accuracy_lgreg_model = accuracy_score(y_true_all_folds, lgreg_model_y_pred)
conf_matrix_lgreg_model = confusion_matrix(y_true_all_folds, lgreg_model_y_pred)
class_report_lgreg_model = classification_report(y_true_all_folds, lgreg_model_y_pred)

