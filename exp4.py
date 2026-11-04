# EXPERIMENT 4: AI FOR MEDICAL PROGNOSIS

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

