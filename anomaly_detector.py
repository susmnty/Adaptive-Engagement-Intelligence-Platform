import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path='user_segments.csv'):
    try:
        df = pd.read_csv(path)
        print("[✓] Loaded user_segments.csv for anomaly detection")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

def preprocess_data(df, features):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    print("[✓] Scaled features for anomaly detection")
    return df_scaled, scaler

def detect_anomalies(df_scaled, contamination=0.05):
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    df_scaled['anomaly_score'] = iso_forest.fit_predict(df_scaled)
    df_scaled['is_anomaly'] = df_scaled['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
    print(f"[✓] Detected {df_scaled['is_anomaly'].sum()} anomalies out of {len(df_scaled)} records")
    return df_scaled, iso_forest

def merge_results(original_df, anomaly_df):
    return pd.concat([original_df.reset_index(drop=True), anomaly_df['is_anomaly']], axis=1)

def save_model(model, scaler, output_path='output/anomaly_model.pkl'):
    joblib.dump({'model': model, 'scaler': scaler}, output_path)
    print(f"[💾] Saved model and scaler as '{output_path}'")

def save_output(df_final, output_csv='output/user_segments_with_anomalies.csv'):
    os.makedirs('output', exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    print(f"[📁] Exported updated file as '{output_csv}'")

def main():
    df = load_data()
    features = ['total_sessions', 'avg_session_length', 'add_to_cart_rate',
                'purchase_rate', 'avg_spend', 'unique_categories']

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in input data: {missing}")

    df_scaled, scaler = preprocess_data(df, features)
    df_anomalies, model = detect_anomalies(df_scaled)
    df_final = merge_results(df, df_anomalies)

    save_output(df_final)
    save_model(model, scaler)

if __name__ == '__main__':
    main()