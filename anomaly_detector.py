# anomaly_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

def load_data(path='user_segments.csv'):
    df = pd.read_csv(path)
    print("[✓] Loaded user_segments.csv for anomaly detection")
    return df

def detect_anomalies(df, features):
    # Use Isolation Forest to find anomalies
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(df[features])
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
    print(f"[✓] Detected {df['is_anomaly'].sum()} anomalies out of {len(df)} records")

    # Save the model
    joblib.dump(iso_forest, 'anomaly_model.pkl')
    print("[💾] Saved model as anomaly_model.pkl")

    return df

def main():
    df = load_data()
    features = ['total_sessions', 'avg_session_length', 'add_to_cart_rate', 
                'purchase_rate', 'avg_spend', 'unique_categories']

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df = detect_anomalies(df, features)
    df.to_csv('user_segments_with_anomalies.csv', index=False)
    print("[📁] Exported updated file as user_segments_with_anomalies.csv")

if __name__ == '__main__':
    main()