import pandas as pd

# Load original depression dataset
df = pd.read_csv("data/raw/depression_dataset.csv")

# Filter only Depressive + Healthy (same as training)
df = df[df["specific.disorder"].isin(["Depressive disorder", "Healthy control"])]

# Drop label columns
drop_cols = ["specific.disorder", "main.disorder"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=col)

# Drop non-numeric columns
df = df.select_dtypes(include=["number"])

# Take first 10 rows for testing
sample_df = df.head(10)

sample_df.to_csv("sample_depression_input.csv", index=False)
print("Sample depression input file created: sample_depression_input.csv")