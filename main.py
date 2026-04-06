
# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. LOAD DATA

train_path = "/kaggle/input/genre-classification-dataset-imdb/train_data.txt"
test_path = "/kaggle/input/genre-classification-dataset-imdb/test_data.txt"
test_solution_path = "/kaggle/input/genre-classification-dataset-imdb/test_data_solution.txt"

# Dataset format: ID ::: TITLE ::: GENRE ::: DESCRIPTION
train_df = pd.read_csv(train_path, sep=" ::: ", engine="python",
                       names=["ID", "TITLE", "GENRE", "DESCRIPTION"])

test_df = pd.read_csv(test_path, sep=" ::: ", engine="python",
                      names=["ID", "TITLE", "DESCRIPTION"])

test_solution_df = pd.read_csv(test_solution_path, sep=" ::: ", engine="python",
                               names=["ID", "TITLE", "GENRE", "DESCRIPTION"])

# 3. DATA PREPROCESSING
X = train_df["DESCRIPTION"]
y = train_df["GENRE"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. TF-IDF VECTORIZATION

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# 5. MODEL TRAINING

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6. VALIDATION

y_pred_val = model.predict(X_val_tfidf)

print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred_val))

# 7. TEST PREDICTION

X_test = test_df["DESCRIPTION"]
X_test_tfidf = tfidf.transform(X_test)

test_predictions = model.predict(X_test_tfidf)

# 8. EVALUATE USING TEST SOLUTION

y_test_actual = test_solution_df["GENRE"]

print("\nTest Accuracy:", accuracy_score(y_test_actual, test_predictions))
print("\nTest Classification Report:\n")
print(classification_report(y_test_actual, test_predictions))


output = pd.DataFrame({
    "ID": test_df["ID"],
    "Predicted_Genre": test_predictions
})

output.to_csv("predictions.csv", index=False)

print("\nPredictions saved to predictions.csv")