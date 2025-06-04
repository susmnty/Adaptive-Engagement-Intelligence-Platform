# dashboard.py
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Engagement Intelligence Dashboard", layout="wide")
st.title("📊 Adaptive Engagement Intelligence Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("user_segments_with_strategies.csv")
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter Users")

segments = df['segment'].unique().tolist()
selected_segments = st.sidebar.multiselect("Select Segments", segments, default=segments)

churn_filter = st.sidebar.selectbox("Churn Status", ["All", "Churned", "Not Churned"])
show_anomalies = st.sidebar.checkbox("Only Show Anomalies", value=False)
user_id = st.sidebar.text_input("Search by User ID")

# Apply Filters
filtered_df = df[df['segment'].isin(selected_segments)]

if churn_filter == "Churned":
    filtered_df = filtered_df[filtered_df['churn'] == 1]
elif churn_filter == "Not Churned":
    filtered_df = filtered_df[filtered_df['churn'] == 0]

if show_anomalies:
    filtered_df = filtered_df[filtered_df['is_anomaly'] == 1]

if user_id:
    filtered_df = filtered_df[filtered_df['user_id'].str.contains(user_id, case=False)]

# Metrics Summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", len(filtered_df))
col2.metric("Churned Users", int(filtered_df['churn'].sum()))
col3.metric("Anomalies", int(filtered_df['is_anomaly'].sum()))
col4.metric("Urgent Outreach Needed", int((filtered_df['retention_strategy'] == 'Urgent outreach').sum()))

# Segment Distribution
st.subheader("User Segment Distribution")
fig1 = px.histogram(filtered_df, x='segment', color='segment', title="User Segments")
st.plotly_chart(fig1, use_container_width=True)

# Retention Strategy Breakdown
st.subheader("Retention Strategy Allocation")
fig2 = px.pie(filtered_df, names='retention_strategy', title='Retention Strategies')
st.plotly_chart(fig2, use_container_width=True)

# Churn vs Anomaly
st.subheader("Churn vs Anomaly Overlap")
fig3 = px.scatter(filtered_df, x="avg_session_length", y="avg_spend", color="churn", 
                  symbol="is_anomaly", hover_data=['retention_strategy'],
                  title="Behavioral Plot: Session Length vs Spend")
st.plotly_chart(fig3, use_container_width=True)

# Data Table
st.subheader("📄 Detailed User Data")
st.dataframe(filtered_df.head(50), use_container_width=True)

st.caption("Built using Streamlit, Plotly, and scikit-learn")