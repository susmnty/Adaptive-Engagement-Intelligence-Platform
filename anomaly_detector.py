# anomaly_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt

def load_data(path='user_segments.csv'):
    df = pd.read_csv(path)
    print("[✓] Loaded user_segments.csv for anomaly detection")
    return df

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

def visualize_anomalies(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    colors = df['is_anomaly'].map({0: 'blue', 1: 'red'})
    plt.scatter(df[x_col], df[y_col], c=colors, alpha=0.6)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Anomaly Detection Visualization')
    plt.grid(True)
    plt.show()

def save_model(model, scaler, output_path='anomaly_model.pkl'):
    joblib.dump({'model': model, 'scaler': scaler}, output_path)
    print(f"[💾] Saved model and scaler as {output_path}")

def main():
    df = load_data()
    features = ['total_sessions', 'avg_session_length', 'add_to_cart_rate', 
                'purchase_rate', 'avg_spend', 'unique_categories']

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df_scaled, scaler = preprocess_data(df, features)
    df_anomalies, model = detect_anomalies(df_scaled)
    df_final = merge_results(df, df_anomalies)

    os.makedirs('output', exist_ok=True)
    df_final.to_csv('output/user_segments_with_anomalies.csv', index=False)
    print("[📁] Exported updated file as output/user_segments_with_anomalies.csv")

    save_model(model, scaler, 'output/anomaly_model.pkl')

    # Optional: visualize anomalies
    visualize_anomalies(df_final, x_col='avg_session_length', y_col='avg_spend')

if __name__ == '__main__':
    main()