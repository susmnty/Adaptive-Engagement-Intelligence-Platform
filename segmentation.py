# cluster.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

FEATURES_CSV = Path("data/user_features.csv")
SEGMENTED_CSV = Path("data/user_segments.csv")

def load_features():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"[❌] {FEATURES_CSV} not found.")
    return pd.read_csv(FEATURES_CSV)

def cluster_users(df, n_clusters=5):
    if "user_id" not in df.columns:
        raise ValueError("[❌] 'user_id' column missing in input features")

    user_ids = df["user_id"]
    X = df.drop(columns=["user_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Suppress future KMeans warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        segments = model.fit_predict(X_scaled)

    df["segment"] = segments
    return df, model

def main():
    df = load_features()
    clustered_df, model = cluster_users(df)
    SEGMENTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(SEGMENTED_CSV, index=False)
    print(f"[✓] User segments written to: {SEGMENTED_CSV}")
    print("\n[📊] Average metrics per segment:")
    print(clustered_df.groupby("segment").mean(numeric_only=True))

if __name__ == "__main__":
    main()