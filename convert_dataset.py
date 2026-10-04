import pandas as pd

files = {
    "train": "dataset/data/train-00000-of-00001.parquet",
    "test": "dataset/data/test-00000-of-00001.parquet",
    "valid": "dataset/data/valid-00000-of-00001.parquet"
}

for name, path in files.items():
    print(f"Reading {name} dataset...")

    df = pd.read_parquet(path)

    output = f"{name}.csv"
    df.to_csv(output, index=False)

    print(f"Saved {output} — {len(df)} rows")

print("\nDataset conversion completed successfully!")