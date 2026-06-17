import pandas as pd

df = pd.read_csv("students_pre_specialization.csv")

# Randomly select 5% of rows
sample_df = df.sample(frac=0.05, random_state=42)

sample_df.to_csv("students_sample.csv", index=False)

print(f"Selected {len(sample_df)} rows.")