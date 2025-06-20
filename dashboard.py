import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ---------- Page Configuration ----------
st.set_page_config(layout="wide")
st.title("Adaptive Engagement IP")

# ---------- Initialize Session State ----------
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'cart' not in st.session_state:
    st.session_state.cart = {}
if 'order_status' not in st.session_state:
    st.session_state.order_status = {}
if 'product_log' not in st.session_state:
    st.session_state.product_log = []

# ---------- Step 1: User Onboarding ----------
st.sidebar.header("User Onboarding")
user_id = st.sidebar.text_input("Enter User ID")
age = st.sidebar.slider("Age", 18, 60)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
location = st.sidebar.selectbox("Location", ['Delhi', 'Mumbai', 'Chennai', 'Hyderabad', 'Bangalore', 'Other'])

if st.sidebar.button("Submit User Info"):
    if user_id:
        st.session_state.users[user_id] = {
            'age': age,
            'gender': gender,
            'location': location,
            'events': []
        }
        st.session_state.cart[user_id] = []
        st.session_state.order_status[user_id] = "browsing"
        st.sidebar.success(f"User {user_id} onboarded successfully!")
    else:
        st.sidebar.warning("Enter valid User ID.")

# ---------- Step 2: Product Browsing ----------
st.header("Browse & Interact with Products")
products = ["Apples", "Rice", "Milk", "Bread", "Toothpaste"]

if user_id in st.session_state.users:
    current_events = st.session_state.users[user_id]['events']
    cart = st.session_state.cart[user_id]
    status = st.session_state.order_status[user_id]

    selected_product = st.selectbox("Choose Product", products)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("View Product Detail"):
            current_events.append("View Product Detail")
            st.session_state.product_log.append((user_id, selected_product, "Viewed", datetime.now()))
            st.success(f"{selected_product} details viewed")

    with col2:
        if st.button("Add to Cart"):
            cart.append(selected_product)
            current_events.append("Add to Cart")
            st.session_state.product_log.append((user_id, selected_product, "Added to Cart", datetime.now()))
            st.success(f"{selected_product} added to cart")

    with col3:
        if cart and status == "browsing" and st.button("Checkout"):
            current_events.append("Checkout")
            st.session_state.order_status[user_id] = "checkout"
            st.session_state.product_log.append((user_id, selected_product, "Checkout", datetime.now()))
            st.success("Proceeding to checkout")

    # Confirm Purchase
    if status == "checkout":
        st.info("You are in checkout stage. Please confirm your purchase.")
        if st.button("Confirm Purchase"):
            current_events.append("Purchase Success")
            st.session_state.order_status[user_id] = "purchased"
            st.session_state.product_log.append((user_id, selected_product, "Purchased", datetime.now()))
            st.success("Order Placed Successfully!")
            st.session_state.cart[user_id] = []  # Clear cart after purchase

    st.markdown("---")
    st.subheader("Your Cart")
    st.write(cart)

    st.subheader("Event Log")
    st.write(current_events)

    st.subheader("Product Interaction Log")
    interaction_df = pd.DataFrame(st.session_state.product_log, columns=["User", "Product", "Action", "Time"])
    interaction_df = interaction_df.sort_values(by="Time", ascending=False)
    st.dataframe(interaction_df)

    # Download as CSV
    csv = interaction_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Interaction Log as CSV",
        data=csv,
        file_name='product_interaction_log.csv',
        mime='text/csv'
    )
else:
    st.info("Please complete user onboarding to proceed.")

# ---------- Step 3: Anomaly Detection ----------
st.header("🚨 Anomalous User Detection")
anomalous_users = []
for uid, info in st.session_state.users.items():
    events = info['events']
    if "Checkout" in events and "Purchase Success" not in events:
        anomalous_users.append(uid)
    elif events.count("Add to Cart") >= 3 and "Checkout" not in events:
        anomalous_users.append(uid)

if anomalous_users:
    st.warning("Anomalous Users Detected:")
    st.write(anomalous_users)
else:
    st.success("No anomalies detected so far.")

# ---------- Step 4: User Segmentation ----------
st.header("User Segmentation")
st.caption(f"Total Users Onboarded: {len(st.session_state.users)}")

def prepare_cluster_data(users):
    rows = []
    for uid, info in users.items():
        row = [uid, info['age']]
        row += [info['events'].count(e) for e in ["View Product Detail", "Add to Cart", "Checkout", "Purchase Success"]]
        rows.append(row)
    df = pd.DataFrame(rows, columns=['user_id', 'age', 'view_detail', 'add_to_cart', 'checkout', 'purchase'])
    return df

if st.button("Run Clustering"):
    df = prepare_cluster_data(st.session_state.users)
    if len(df) >= 1:
        features = df.drop('user_id', axis=1)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        n_clusters = min(3, len(df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled)

        st.subheader("Clustered Users")
        st.dataframe(df)

        scatter_fig = px.scatter(df, x='age', y='purchase', color='cluster', hover_data=['user_id'])
        st.plotly_chart(scatter_fig)

        cluster_counts = df['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'User Count']
        pie_fig = px.pie(cluster_counts, names='Cluster', values='User Count', title="User Distribution by Cluster")
        st.plotly_chart(pie_fig)
    else:
        st.warning(f"Add at least 1 user to perform clustering. Currently added: {len(df)}")