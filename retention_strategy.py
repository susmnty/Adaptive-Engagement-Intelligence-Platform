from pathlib import Path
import pandas as pd

INPUT_CSV = Path("user_segments_with_churn_and_anomalies.csv")
OUTPUT_CSV = Path("user_segments_with_strategies.csv")

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"[❌] Input file '{path}' not found.")
    df = pd.read_csv(path)
    print(f"[✓] Loaded data: {len(df)} rows")
    return df

def assign_retention_strategy(row):
    churn = row.get('churn')
    anomaly = row.get('is_anomaly')

    if pd.isna(churn) or pd.isna(anomaly):
        return "Incomplete data"

    if churn == 1 and anomaly == 1:
        return "Urgent outreach"
    elif churn == 1:
        return "Incentive offer"
    elif anomaly == 1:
        return "Behavior review"
    else:
        return "Maintain engagement"

def main():
    df = load_data(INPUT_CSV)

    required_cols = {'churn', 'is_anomaly'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[❌] Required columns missing: {missing}")

    df['retention_strategy'] = df.apply(assign_retention_strategy, axis=1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[📁] Output saved to: {OUTPUT_CSV}")
    print("\n[📊] Strategy distribution:\n", df['retention_strategy'].value_counts())

if __name__ == "__main__":
    main()