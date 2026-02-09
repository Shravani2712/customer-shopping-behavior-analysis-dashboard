import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Customer Behavior Analysis Dashboard",
    layout="wide"
)

st.title("🛍️ Customer Shopping Behavior Analysis")
st.markdown(
    "Interactive dashboard built from customer shopping behavior data "
    "to analyze purchasing patterns, revenue, demographics, and geography."
)

# ------------------ FILE UPLOAD ------------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload customer_shopping_behavior.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⬅️ Please upload the customer shopping behavior CSV file.")
    st.stop()

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

# ------------------ DATA CLEANING ------------------
df["review_rating"] = df.groupby("category")["review_rating"].transform(
    lambda x: x.fillna(x.median())
)

labels = ["Young Adult", "Adult", "Middle-aged", "Senior"]
df["age_group"] = pd.qcut(df["age"], q=4, labels=labels)

frequency_mapping = {
    "Fortnightly": 14,
    "Weekly": 7,
    "Monthly": 30,
    "Quarterly": 90,
    "Bi-Weekly": 14,
    "Annually": 365,
    "Every 3 Months": 90
}

df["purchase_frequency_days"] = df["frequency_of_purchases"].map(frequency_mapping)

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.header("🔍 Filters")

selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["category"].unique())
)

selected_gender = st.sidebar.selectbox(
    "Select Gender",
    ["All"] + sorted(df["gender"].unique())
)

filtered_df = df.copy()

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender]

# ------------------ KPI SECTION ------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Customers", filtered_df["customer_id"].nunique())
col2.metric("Total Revenue", f"{filtered_df['purchase_amount'].sum():,.0f}")
col3.metric("Avg Purchase Value", round(filtered_df["purchase_amount"].mean(), 2))
col4.metric("Top Category", filtered_df["category"].mode()[0])
col5.metric("Top Payment Method", filtered_df["payment_method"].mode()[0])

# ------------------ BAR CHARTS ------------------
st.subheader("📊 Purchase Analysis")

col6, col7, col8 = st.columns(3)

with col6:
    fig, ax = plt.subplots()
    filtered_df.groupby("category")["purchase_amount"].sum().plot(kind="bar", ax=ax)
    ax.set_title("Purchase Amount by Category")
    st.pyplot(fig)

with col7:
    fig, ax = plt.subplots()
    filtered_df["gender"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Purchases by Gender")
    st.pyplot(fig)

with col8:
    fig, ax = plt.subplots()
    filtered_df["payment_method"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Payment Method Usage")
    st.pyplot(fig)

# ------------------ MORE VISUALS ------------------
col9, col10, col11 = st.columns(3)

with col9:
    fig, ax = plt.subplots()
    filtered_df.groupby("season")["purchase_amount"].sum().plot(kind="bar", ax=ax)
    ax.set_title("Season-wise Sales")
    st.pyplot(fig)

with col10:
    fig, ax = plt.subplots()
    filtered_df["discount_applied"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Discount Applied vs Not Applied")
    st.pyplot(fig)

with col11:
    fig, ax = plt.subplots()
    ax.hist(filtered_df["age"], bins=10)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    st.pyplot(fig)

# ------------------ HEATMAP ------------------
st.subheader("🔥 Category vs Age Group Heatmap")

heatmap_data = pd.pivot_table(
    filtered_df,
    values="purchase_amount",
    index="category",
    columns="age_group",
    aggfunc="sum",
    fill_value=0
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".0f", ax=ax)
st.pyplot(fig)

# ------------------ MAP VISUALIZATION ------------------
st.subheader("🌍 Customer Distribution by Location")

location_counts = (
    filtered_df.groupby("location")
    .size()
    .reset_index(name="customer_count")
)

fig_map = px.scatter_geo(
    location_counts,
    locations="location",
    locationmode="country names",
    size="customer_count",
    hover_name="location",
    projection="natural earth",
    title="Customers by Country",
    size_max=40
)

st.plotly_chart(fig_map, use_container_width=True)

# ------------------ DATA TABLE ------------------
st.subheader("📄 Filtered Dataset")
st.dataframe(filtered_df)