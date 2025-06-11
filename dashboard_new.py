import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
import os

# File to persist data
DATA_FILE = "events.csv"

# Load existing data or initialize
if 'event_data' not in st.session_state:
    if os.path.exists(DATA_FILE):
        st.session_state.event_data = pd.read_csv(DATA_FILE)
    else:
        st.session_state.event_data = pd.DataFrame(columns=['username', 'product_id', 'event'])

# Event mapping to numerical value for clustering
event_order = {
    'Product List': 0,
    'Product Detail': 1,
    'Add to Cart': 2,
    'Checkout': 3,
    'Success': 4
}

# Cluster interpretation labels
cluster_labels = {
    0: "Window Shopper",
    1: "Interested but Didn’t Buy",
    2: "Cart Abandoner",
    3: "Stuck at Checkout",
    4: "Satisfied Customer"
}

# LEFT PANEL: Input Form
st.sidebar.title("📋 Register Event")
username = st.sidebar.text_input("Username", "user01")
product_id = st.sidebar.text_input("Product ID", "P123")
event = st.sidebar.selectbox("Select Event", list(event_order.keys()))
submit = st.sidebar.button("Register Event")

# Add event to data and save to CSV
if submit:
    new_event = pd.DataFrame([[username, product_id, event]], columns=['username', 'product_id', 'event'])
    st.session_state.event_data = pd.concat([st.session_state.event_data, new_event], ignore_index=True)
    st.session_state.event_data.to_csv(DATA_FILE, index=False)
    st.success("✅ Event registered and saved!")

df = st.session_state.event_data.copy()

# Process for clustering
if not df.empty:
    df['event_num'] = df['event'].map(event_order)
    event_matrix = pd.crosstab(df['username'], df['event'])

    # Apply clustering
    k = min(5, len(event_matrix))  # Avoid more clusters than users
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(event_matrix)
    event_matrix['Cluster'] = clusters

    # Display Cluster Distribution
    st.title("🧠 User Behavior Clustering")
    cluster_counts = event_matrix['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    cluster_counts['Cluster Name'] = cluster_counts['Cluster'].map(cluster_labels)
    fig = px.pie(cluster_counts, names='Cluster Name', values='Count', title="Cluster Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Display Users by Cluster
    st.header("👥 Users by Cluster")
    for cluster_id in sorted(event_matrix['Cluster'].unique()):
        users = event_matrix[event_matrix['Cluster'] == cluster_id].index.tolist()
        st.subheader(f"🟢 {cluster_labels.get(cluster_id, 'Unknown')}")
        st.write(users)

else:
    st.warning("⚠️ No event data yet. Register some events to see clustering results.")
