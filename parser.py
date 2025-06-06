# parse_logs.py
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

RAW_LOGS = Path("data/raw_logs.jsonl")
STRUCTURED_CSV = Path("data/structured_data.csv")
SESSION_TIMEOUT = timedelta(minutes=30)

def load_logs():
    data = []
    with open(RAW_LOGS, 'r') as f:
        for line in f:
            event = json.loads(line.strip())
            meta = event.get("meta", {})  # Safely handle missing meta field
            row = {
                "user_id": event.get("uid"),
                "event": event.get("event"),
                "timestamp": datetime.fromisoformat(event.get("time")),
                "product_id": meta.get("product_id"),
                "category": meta.get("category"),
                "price": meta.get("price")
            }
            data.append(row)
    return pd.DataFrame(data)

def assign_sessions(df):
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
    session_ids = []
    last_time = {}
    session_count = {}

    for i, row in df.iterrows():
        uid = row["user_id"]
        curr_time = row["timestamp"]

        if uid not in last_time or (curr_time - last_time[uid]) > SESSION_TIMEOUT:
            session_count[uid] = session_count.get(uid, 0) + 1

        session_id = f"{uid}_sess_{session_count[uid]}"
        session_ids.append(session_id)
        last_time[uid] = curr_time

    df["session_id"] = session_ids
    return df

def main():
    df = load_logs()
    df = assign_sessions(df)
    df["timestamp"] = df["timestamp"].dt.isoformat()  # Optional: Make timestamp string for CSV
    df.to_csv(STRUCTURED_CSV, index=False)
    print(f"[✓] Parsed {len(df)} log events")
    print(f"[✓] Structured CSV written to: {STRUCTURED_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()