import joblib

# Load model and vectorizer
model = joblib.load("bias_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Mapping from dataset labels
bias_labels = {
    0: "Bias Class 0",
    1: "Bias Class 1",
    2: "Bias Class 2"
}

print("===================================")
print("   POLITICAL BIAS CLASSIFIER")
print("===================================")

article = input("\nEnter an article or text:\n")

# Convert text into TF-IDF
text_vector = vectorizer.transform([article])

# Predict
prediction = model.predict(text_vector)[0]

# Display result
print("\n-----------------------------------")
print("Predicted Bias:", bias_labels.get(prediction, prediction))
print("-----------------------------------")