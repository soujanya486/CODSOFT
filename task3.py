
# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 2. LOAD DATA

df = pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv")

print(df.head())

# 3. DATA PREPROCESSING

# Drop unnecessary columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors='ignore')

# Encode categorical variables
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Geography"] = le.fit_transform(df["Geography"])

# 4. DEFINE FEATURES & TARGET

X = df.drop("Exited", axis=1)
y = df["Exited"]

# 5. FEATURE SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. MODEL TRAINING

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# 8. EVALUATION FUNCTION

def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 9. EVALUATE MODELS

evaluate(lr, "Logistic Regression")
evaluate(rf, "Random Forest")
evaluate(gb, "Gradient Boosting")

# 10. SAVE BEST MODEL

import joblib
joblib.dump(rf, "churn_model.pkl")

print("\nModel saved as churn_model.pkl")