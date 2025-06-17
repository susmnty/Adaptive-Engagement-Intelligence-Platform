import pandas as pd
from pathlib import Path

INPUT_CSV = Path("user_segments_with_churn_and_anomalies.csv")
OUTPUT_CSV = Path("user_segments_with_strategies.csv")

def load_data(path=INPUT_CSV):
    if not path.exists():
        raise FileNotFoundError(f"[❌] Input file '{path}' not found.")
    df = pd.read_csv(path)
    print("[✓] Loaded data with churn and anomaly labels")
    return df

def assign_retention_strategy(row):
    if row['churn'] == 1 and row['is_anomaly'] == 1:
        return "Urgent outreach"
    elif row['churn'] == 1:
        return "Incentive offer"
    elif row['is_anomaly'] == 1:
        return "Behavior review"
    else:
        return "Maintain engagement"

def main():
    df = load_data()

    # Ensure required columns exist
    required_cols = {'churn', 'is_anomaly'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[❌] Required columns missing: {required_cols - set(df.columns)}")

    # Apply strategy logic
    df['retention_strategy'] = df.apply(assign_retention_strategy, axis=1)

    # Save updated file
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[📁] Saved output with strategies to: {OUTPUT_CSV}")
    print("\n[📊] Strategy distribution:\n", df['retention_strategy'].value_counts())

if __name__ == "__main__":
    main()