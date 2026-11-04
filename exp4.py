# EXPERIMENT 4: AI FOR MEDICAL PROGNOSIS
# if seaborn not installed then install the seaborn as python -m pip install seaborn
# 1. Software Tools Used

#1. Python 3: Programming language used to implement the experiment.
#2. Google Colab: Platform used to write and execute Python code.
#3. NumPy: Used for numerical calculations.
#4. Pandas: Used to create and manage the patient dataset.
#5. Matplotlib: Used to plot ROC curves and feature importance graphs.
#6. Seaborn: Used for data visualization.
#7. Scikit-learn: Used for data preprocessing, Random Forest classification and model evaluation.

#2. Theory (Short)

#Medical prognosis is the process of predicting the future outcome or progress of a disease. It helps doctors understand the chances of patient recovery or mortality. Unlike diagnosis, which identifies a disease, prognosis predicts the possible outcome of a disease.

#In this experiment, a sample ICU patient dataset is created using clinical features such as age, blood pressure, heart rate, oxygen level and comorbidity score.

#The data is standardized and divided into training and testing sets. A Random Forest algorithm is used to predict patient outcomes (Survived or Died).

#The model is evaluated using a classification report, confusion matrix and ROC-AUC score. Feature importance is also used to identify the most influential clinical factors.

#3. Conclusion
#Thus, AI for Medical Prognosis was successfully implemented using the Random Forest algorithm. Patient outcomes were predicted based on clinical features, and model performance was evaluated using ROC-AUC, classification report and confusion matrix. Oxygen level was identified as the most important feature influencing patient prognosis.

# STEP 1: Import general libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2: Import Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# STEP 3: Create sample ICU prognosis dataset
data = pd.DataFrame({
    'age': [70, 55, 80, 45, 67, 50, 76, 33, 60, 85],
    'blood_pressure': [120, 140, 110, 100, 90, 150, 130, 110, 125, 115],
    'heart_rate': [80, 95, 70, 85, 110, 100, 105, 72, 88, 76],
    'oxygen_level': [92, 88, 94, 96, 85, 90, 87, 98, 91, 93],
    'comorbidity_score': [2, 3, 4, 1, 5, 2, 3, 0, 2, 4],
    'outcome': [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
})

print("Sample ICU Dataset:")
print(data)

# STEP 4: Separate features and target
X = data.drop('outcome', axis=1)
y = data['outcome']

# STEP 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# STEP 7: Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

# STEP 8: Evaluate the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", round(roc_auc, 2))

# STEP 9: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - ICU Mortality Prediction")
plt.legend()
plt.show()

# STEP 10: Plot Feature Importance
importances = model.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance in Prognosis")
plt.show()

print("\nExperiment Completed Successfully!")

