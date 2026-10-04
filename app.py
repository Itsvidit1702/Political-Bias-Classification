import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("bias_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Load Dataset
# -----------------------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Create label mapping
label_mapping = (
    train[["bias", "bias_text"]]
    .drop_duplicates()
    .set_index("bias")["bias_text"]
    .to_dict()
)

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Political Bias Classifier",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Political Bias Classification")
st.write(
    "Machine Learning system for classifying the political bias "
    "of news articles using TF-IDF and Logistic Regression."
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Prediction", "Model Evaluation"]
)

# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "Prediction":

    st.header("🔍 Article Prediction")

    article = st.text_area(
        "Enter News Article",
        height=300,
        placeholder="Paste a news article here..."
    )

    if st.button("Analyze Article"):

        if not article.strip():

            st.warning("Please enter an article.")

        else:

            # TF-IDF transformation
            text_vector = vectorizer.transform([article])

            # Prediction
            prediction = model.predict(text_vector)[0]

            # Probabilities
            probabilities = model.predict_proba(text_vector)[0]

            # Label
            label_name = label_mapping.get(
                prediction,
                f"Class {prediction}"
            )

            st.subheader("Prediction")

            st.success(
                f"Predicted Political Bias: **{label_name}**"
            )

            st.subheader("Prediction Confidence")

            for label, probability in zip(
                model.classes_,
                probabilities
            ):

                display_name = label_mapping.get(
                    label,
                    f"Class {label}"
                )

                st.write(
                    f"**{display_name}: {probability:.2%}**"
                )

                st.progress(float(probability))


# ============================================================
# MODEL EVALUATION PAGE
# ============================================================

else:

    st.header("📊 Model Evaluation")

    test = test.dropna(subset=["content"])

    X_test = test["content"]
    y_test = test["bias"]

    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Predictions
    predictions = model.predict(X_test_tfidf)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Metrics
    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    macro_f1 = report["macro avg"]["f1-score"]

    col2.metric(
        "Macro F1",
        f"{macro_f1 * 100:.2f}%"
    )

    total_articles = len(test)

    col3.metric(
        "Test Articles",
        total_articles
    )

    # -----------------------------
    # Classification Report
    # -----------------------------

    st.subheader("Classification Report")

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Political Bias Confusion Matrix")

    ax.set_xticks(range(len(model.classes_)))
    ax.set_yticks(range(len(model.classes_)))

    ax.set_xticklabels(model.classes_)
    ax.set_yticklabels(model.classes_)

    # Add numbers
    for i in range(len(cm)):
        for j in range(len(cm[i])):

            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)