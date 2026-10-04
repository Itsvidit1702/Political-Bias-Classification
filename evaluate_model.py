import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load model and vectorizer
model = joblib.load("bias_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load test data
test = pd.read_csv("test.csv")

test = test.dropna(subset=["content"])

X_test = test["content"]
y_test = test["bias"]

# Convert text to TF-IDF
X_test_tfidf = vectorizer.transform(X_test)

# Predict
predictions = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print(" MODEL EVALUATION")
print("==============================")

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

# Confusion matrix
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))