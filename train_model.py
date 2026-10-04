import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load datasets
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test = pd.read_csv("test.csv")

# Remove rows with missing text
train = train.dropna(subset=["content"])
valid = valid.dropna(subset=["content"])
test = test.dropna(subset=["content"])

# Input text
X_train = train["content"]
X_valid = valid["content"]
X_test = test["content"]

# Target
y_train = train["bias"]
y_valid = valid["bias"]
y_test = test["bias"]

# TF-IDF
print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")

# Train model
print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Training completed!")

# Validation
valid_predictions = model.predict(X_valid_tfidf)

print("\n--- Validation Results ---")
print("Accuracy:", accuracy_score(y_valid, valid_predictions))
print(classification_report(y_valid, valid_predictions))

# Test
test_predictions = model.predict(X_test_tfidf)

print("\n--- Test Results ---")
print("Accuracy:", accuracy_score(y_test, test_predictions))
print(classification_report(y_test, test_predictions))

# Save model
joblib.dump(model, "bias_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")