import pandas as pd
from hmmlearn import hmm
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load event data
df = pd.read_csv("structured_data.csv")

# Encode event types to integers
event_encoder = LabelEncoder()
df['event_code'] = event_encoder.fit_transform(df['event'])

# Sort data for consistent sequences
df = df.sort_values(by=['user_id', 'timestamp'])

# Group by user and product
sequences = []
lengths = []

for _, group in df.groupby(['user_id', 'product_id']):
    event_seq = group['event_code'].tolist()
    if len(event_seq) > 1:  # HMM needs sequences with at least 2 steps
        sequences.extend(event_seq)
        lengths.append(len(event_seq))

# Reshape for hmmlearn
X = np.array(sequences).reshape(-1, 1)

# Train an HMM model
model = hmm.MultinomialHMM(n_components=4, n_iter=100, random_state=42)
model.fit(X, lengths)

print("✅ HMM model trained successfully on user-event sequences.")