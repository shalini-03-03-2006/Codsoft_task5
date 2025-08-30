# Codsoft_task5
Excellent ⚡ — Fraud detection is one of the most important classification tasks in machine learning.
The Credit Card Fraud Detection dataset (Kaggle) is widely used.

This dataset is highly imbalanced (fraudulent transactions are <1%).
We’ll handle this using SMOTE oversampling or undersampling.


---

📝 Step-by-Step: Credit Card Fraud Detection (Classification)

1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

from imblearn.over_sampling import SMOTE


---

2. Load Dataset

# Load dataset (download creditcard.csv from Kaggle)
data = pd.read_csv("creditcard.csv")

print(data.head())
print(data.info())
print(data["Class"].value_counts())  # 0 = genuine, 1 = fraud


---

3. Preprocessing

Normalize Amount & Time.

Features V1..V28 are PCA-transformed (no need for encoding).

Target = Class.


# Scale 'Amount' and 'Time'
scaler = StandardScaler()
data["Amount"] = scaler.fit_transform(data[["Amount"]])
data["Time"] = scaler.fit_transform(data[["Time"]])

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]


---

4. Handle Class Imbalance (SMOTE Oversampling)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())


---

5. Train Classification Models

Logistic Regression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)


---

6. Evaluate Models

# Logistic Regression
print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))

# Random Forest
print("\nRandom Forest Results")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# Confusion matrix for Random Forest
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.title("Confusion Matrix - Random Forest")
plt.show()


---

✅ End Result:

Logistic Regression → good for interpretability.

Random Forest → usually higher recall (catching fraud cases).

SMOTE balances the dataset to improve fraud detection.

Metrics: Precision, Recall, F1-score, ROC-AUC give a complete picture.



---

👉 Do you want me to also add a comparison table of both models (Logistic vs Random Forest) so you can see which performs better on fraud detection?

