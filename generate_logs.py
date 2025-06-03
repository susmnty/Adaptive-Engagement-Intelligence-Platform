# generate_logs.py
import json
import random
from datetime import datetime, timedelta
import os

# Constants
NUM_USERS = 1000
NUM_PRODUCTS = 500
NUM_EVENTS = 500000
EVENT_TYPES = ["view_product", "click", "add_to_cart", "purchase", "wishlist", "remove", "hover", "scroll", "review"]
CATEGORIES = ["electronics", "books", "fashion", "home", "toys", "sports"]
RAW_LOGS_PATH = "data/raw_logs.jsonl"

# Prepare directories
os.makedirs("data", exist_ok=True)

# Generate IDs
user_ids = [f"user_{i}" for i in range(1, NUM_USERS + 1)]
product_ids = [f"product_{i}" for i in range(1, NUM_PRODUCTS + 1)]

# Generate logs
start_time = datetime(2025, 6, 1)
with open(RAW_LOGS_PATH, "w") as f:
    for _ in range(NUM_EVENTS):
        timestamp = start_time + timedelta(seconds=random.randint(0, 86400 * 7))
        uid = random.choice(user_ids)
        event = random.choice(EVENT_TYPES)
        meta = {
            "product_id": random.choice(product_ids),
            "category": random.choice(CATEGORIES),
            "price": round(random.uniform(5, 500), 2) if event == "purchase" else None
        }
        log = {"uid": uid, "event": event, "time": timestamp.isoformat(), "meta": meta}
        f.write(json.dumps(log) + "\n")

print(f"[✓] Raw logs generated: {RAW_LOGS_PATH}")