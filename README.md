# 📰 Political Bias Classification

A machine learning project for classifying the political bias of news articles using **TF-IDF** and **Logistic Regression**.

## 📌 Project Overview

This project takes the text of a news article as input and predicts its political bias category.

The system uses:

* **TF-IDF (Term Frequency–Inverse Document Frequency)** for text feature extraction
* **Logistic Regression** for classification
* **Scikit-learn** for machine learning
* **Pandas** for dataset processing
* **Streamlit** for the interactive web application

## 📂 Dataset

The project uses the **Article Bias Prediction Random Splits** dataset from Hugging Face.

Dataset repository:

`siddharthmb/article-bias-prediction-random-splits`

The dataset contains separate training, validation, and testing splits.

| Split      | Number of Articles |
| ---------- | -----------------: |
| Training   |             27,978 |
| Validation |              6,996 |
| Testing    |              1,300 |

The dataset contains information such as:

* Topic
* Source
* Bias
* Article title
* Date
* Authors
* Article content
* Bias text

## ⚙️ Machine Learning Pipeline

```text
News Article
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Bias Classification
      ↓
Prediction + Confidence
```

## 🗂️ Project Structure

```text
Political-Bias-Classification/
│
├── dataset/
│   └── data/
│
├── train.csv
├── valid.csv
├── test.csv
│
├── train_model.py
├── predict.py
├── evaluate_model.py
├── plot_confusion_matrix.py
├── convert_dataset.py
├── check_dataset.py
│
├── app.py
│
├── bias_classifier.pkl
├── tfidf_vectorizer.pkl
├── confusion_matrix.png
│
└── README.md
```

## 🚀 Installation

Clone the repository and install the required packages:

```bash
pip install pandas scikit-learn joblib streamlit matplotlib seaborn pyarrow
```

## ▶️ Train the Model

Run:

```bash
python train_model.py
```

This creates:

```text
bias_classifier.pkl
tfidf_vectorizer.pkl
```

## 🔮 Make a Prediction

Run:

```bash
python predict.py
```

Enter a news article when prompted.

## 🌐 Run the Web Application

Start Streamlit:

```bash
python -m streamlit run app.py
```

The application provides:

* Article prediction
* Prediction confidence
* Model accuracy
* Classification report
* Confusion matrix

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The test set contains **1,300 articles**.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib
* Streamlit
* Matplotlib
* Seaborn
* Hugging Face Datasets

## ⚠️ Disclaimer

This project is an educational machine-learning system. Its predictions represent model outputs and should not be treated as definitive judgments about the political orientation or reliability of a news source or article.
