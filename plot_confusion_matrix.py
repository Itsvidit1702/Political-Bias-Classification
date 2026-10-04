import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# Load model and vectorizer
model = joblib.load("bias_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load test dataset
test = pd.read_csv("test.csv")
test = test.dropna(subset=["content"])

X_test = test["content"]
y_test = test["bias"]

# Transform text
X_test_tfidf = vectorizer.transform(X_test)

# Predictions
predictions = model.predict(X_test_tfidf)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Political Bias Classification - Confusion Matrix")

plt.tight_layout()

# Save image
plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

print("Confusion matrix saved as confusion_matrix.png")