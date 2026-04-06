
# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 2. LOAD DATA

df = pd.read_csv("/kaggle/input/fraud-detection/fraudTrain.csv")

# Drop unnecessary column
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

print(df.head())

# 3. DATA PREPROCESSING

# Convert categorical columns to numeric
categorical_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 4. DEFINE FEATURES & TARGET

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. MODEL TRAINING

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 7. EVALUATION FUNCTION

def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 8. EVALUATE ALL MODELS

evaluate(lr, "Logistic Regression")
evaluate(dt, "Decision Tree")
evaluate(rf, "Random Forest")

# 9. BEST MODEL (OPTIONAL SAVE)

import joblib
joblib.dump(rf, "fraud_model.pkl")

print("\nModel saved as fraud_model.pkl")