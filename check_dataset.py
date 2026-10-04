import pandas as pd

train = pd.read_csv("train.csv")

print("\n--- Dataset Shape ---")
print(train.shape)

print("\n--- Columns ---")
print(train.columns.tolist())

print("\n--- First 5 Rows ---")
print(train.head())

print("\n--- Missing Values ---")
print(train.isnull().sum())

print("\n--- Bias Distribution ---")
print(train["bias"].value_counts())

print("\n--- Bias Text Distribution ---")
print(train["bias_text"].value_counts())