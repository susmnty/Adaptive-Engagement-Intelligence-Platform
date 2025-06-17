# features.py
import pandas as pd
from pathlib import Path

STRUCTURED_CSV = Path("data/structured_data.csv")
FEATURES_CSV = Path("data/user_features.csv")

def load_data():
    return pd.read_csv(STRUCTURED_CSV, parse_dates=["timestamp"])

def extract_features(df):
    features = []
    for user_id, group in df.groupby("user_id"):
        session_counts = group["session_id"].nunique()
        
        # Get session durations
        session_lengths = group.groupby("session_id")["timestamp"].agg(["min", "max"])
        session_durations = (session_lengths["max"] - session_lengths["min"]).dt.total_seconds() / 60
        avg_session_length = session_durations.mean() if not session_durations.empty else 0

        total_events = len(group)
        add_to_cart = (group["event"] == "add_to_cart").sum()
        purchase = (group["event"] == "purchase").sum()

        add_to_cart_rate = add_to_cart / total_events if total_events else 0
        purchase_rate = purchase / total_events if total_events else 0
        
        # Safely calculate average spend
        purchases = group[group["event"] == "purchase"]
        avg_spend = purchases["price"].mean() if not purchases.empty else 0
        
        unique_categories = group["category"].nunique()

        features.append({
            "user_id": user_id,
            "total_sessions": session_counts,
            "avg_session_length": round(avg_session_length, 2),
            "add_to_cart_rate": round(add_to_cart_rate, 2),
            "purchase_rate": round(purchase_rate, 2),
            "avg_spend": round(avg_spend, 2),
            "unique_categories": unique_categories
        })

    return pd.DataFrame(features)

def main():
    df = load_data()
    user_features = extract_features(df)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    user_features.to_csv(FEATURES_CSV, index=False)
    print(f"[✓] User features written to: {FEATURES_CSV}")
    print(user_features.head())

if __name__ == "__main__":
    main()