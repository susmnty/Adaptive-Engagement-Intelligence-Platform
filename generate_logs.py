import json
import random
from datetime import datetime, timedelta
import os

# Constants
NUM_USERS = 500
NUM_PRODUCTS = 300
NUM_EVENTS = 100000
EVENT_TYPES = ["view_product", "click", "add_to_cart", "purchase", "wishlist", "remove", "hover", "scroll"]
CATEGORIES = ["electronics", "books", "fashion", "home", "toys", "sports"]
RAW_LOGS_PATH = "data/raw_logs.jsonl"

# Prepare directory
os.makedirs("data", exist_ok=True)

# Generate IDs
user_ids = [f"user_{i}" for i in range(1, NUM_USERS + 1)]
product_ids = [f"product_{i}" for i in range(1, NUM_PRODUCTS + 1)]

# User behavior personas
user_personas = {
    "window_shopper": 0.3,
    "active_buyer": 0.25,
    "impulse_buyer": 0.2,
    "cart_abandoner": 0.15,
    "loyal_browser": 0.1
}

persona_distribution = random.choices(
    population=list(user_personas.keys()),
    weights=list(user_personas.values()),
    k=NUM_USERS
)

user_persona_map = {uid: persona for uid, persona in zip(user_ids, persona_distribution)}

# Helper: Generate timestamp with time-of-day and weekday weight
def generate_realistic_timestamp(start_date):
    base_day = start_date + timedelta(days=random.randint(0, 6))
    hour = random.choices(
        population=[10, 13, 16, 19, 22],
        weights=[1, 2, 3, 5, 2]
    )[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(base_day.year, base_day.month, base_day.day, hour, minute, second)

# Event sequences per persona
persona_events = {
    "window_shopper": ["view_product", "scroll", "hover"],
    "active_buyer": ["view_product", "click", "add_to_cart", "purchase"],
    "impulse_buyer": ["click", "add_to_cart", "purchase"],
    "cart_abandoner": ["view_product", "add_to_cart", "remove"],
    "loyal_browser": ["view_product", "wishlist", "scroll"]
}

# Log generation
start_date = datetime(2025, 6, 1)

with open(RAW_LOGS_PATH, "w") as f:
    for _ in range(NUM_EVENTS):
        uid = random.choice(user_ids)
        persona = user_persona_map[uid]
        timestamp = generate_realistic_timestamp(start_date)
        event = random.choice(persona_events[persona])

        meta = {
            "product_id": random.choice(product_ids),
            "category": random.choice(CATEGORIES),
            "price": round(random.uniform(10, 300), 2) if event == "purchase" else None
        }

        log = {
            "uid": uid,
            "event": event,
            "time": timestamp.isoformat(),
            "meta": meta
        }

        f.write(json.dumps(log) + "\n")

print(f"[✓] Realistic raw logs generated: {RAW_LOGS_PATH}")