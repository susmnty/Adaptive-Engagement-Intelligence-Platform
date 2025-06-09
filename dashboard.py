import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(layout="wide")
st.title("Adaptive Engagement Intelligence Dashboard")

# ---------- Load or simulate event data ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("user_events.csv")  # Your real data file
    except FileNotFoundError:
        # Simulated fallback data
        df = pd.DataFrame({
            "user_id": np.random.choice([f"user{i}" for i in range(1, 51)], 500),
            "event": np.random.choice(
                ["product list visit", "visit product details", "add to cart", "checkout", "success"],
                500,
                p=[0.3, 0.25, 0.2, 0.15, 0.1]
            ),
            "product_id": np.random.randint(100, 110, 500)
        })
    return df

data = load_data()

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.header("User Info & Event Input")

    username = st.text_input("Username")
    user_id = st.text_input("User ID")
    product_id = st.text_input("Product ID")
    event_type = st.selectbox("Event Type", [
        "product list visit", "visit product details", "add to cart", "checkout", "success"
    ])

    if st.button("Register Event"):
        if username and user_id and product_id:
            new_row = pd.DataFrame([{
                "user_id": user_id,
                "event": event_type,
                "product_id": product_id
            }])
            data = pd.concat([data, new_row], ignore_index=True)
            st.success(f"Event '{event_type}' registered for user {user_id}")
        else:
            st.warning("Please fill all fields to register an event.")

# ---------- Funnel Stage Chart ----------
st.subheader("Funnel Stage Distribution")

funnel_steps = ["product list visit", "visit product details", "add to cart", "checkout", "success"]
funnel_counts = {
    step: data[data["event"] == step]["user_id"].nunique() for step in funnel_steps
}
funnel_df = pd.DataFrame({
    "Stage": list(funnel_counts.keys()),
    "Users": list(funnel_counts.values())
})

fig_funnel = px.bar(
    funnel_df,
    x="Stage",
    y="Users",
    color="Stage",
    title="User Distribution Across Engagement Stages",
    text="Users"
)
fig_funnel.update_traces(textposition="outside")
st.plotly_chart(fig_funnel, use_container_width=True)

# ---------- Show Dataset for Specific User ----------
if user_id:
    user_events = data[data['user_id'] == user_id]
    st.subheader(f"Events for User ID: `{user_id}`")
    st.dataframe(user_events, use_container_width=True)
else:
    st.info("Enter a User ID above to display their event history.")

# ---------- Clustering Section ----------
st.subheader("User Clustering (5 Segments)")

pivot_df = data.pivot_table(index="user_id", columns="event", aggfunc="size", fill_value=0)
pivot_df_std = StandardScaler().fit_transform(pivot_df)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pivot_df_std)

pivot_df["cluster"] = clusters

# Cluster labeling (customizable)
cluster_labels = {
    0: "Window Shopper",
    1: "Interested but Didn’t Buy",
    2: "Cart Abandoner",
    3: "Stuck at Checkout",
    4: "Satisfied Customer"
}
pivot_df["cluster_label"] = pivot_df["cluster"].map(cluster_labels)

# Cluster pie chart
cluster_summary = pivot_df["cluster"].value_counts().reset_index()
cluster_summary.columns = ["Cluster", "Users"]
cluster_summary["Label"] = cluster_summary["Cluster"].map(cluster_labels)

fig_cluster = px.pie(
    cluster_summary,
    names="Label",
    values="Users",
    title="User Cluster Distribution"
)
st.plotly_chart(fig_cluster, use_container_width=True)

# Display users by cluster
st.markdown("### 👥 Users by Cluster")
for cluster_id in sorted(pivot_df["cluster"].unique()):
    users_in_cluster = pivot_df[pivot_df["cluster"] == cluster_id].index.tolist()
    label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
    st.markdown(f"**Cluster {cluster_id} - {label}**")
    st.json(users_in_cluster)