import pandas as pd

def load_data(path='user_segments_with_churn_and_anomalies.csv'):
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

    # Check necessary columns
    if 'churn' not in df.columns or 'is_anomaly' not in df.columns:
        raise ValueError("Required columns 'churn' or 'is_anomaly' not found.")

    # Apply strategy logic
    df['retention_strategy'] = df.apply(assign_retention_strategy, axis=1)

    # Save updated file
    df.to_csv('user_segments_with_strategies.csv', index=False)
    print("[📁] Saved output to user_segments_with_strategies.csv")

if __name__ == "__main__":
    main()