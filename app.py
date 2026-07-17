"""Car Selling Price Predictor — uses bundled car.csv and pre-trained LinearRegression."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).parent
DATA_PATH = BASE / "car.csv"
MODEL_PATH = BASE / "car_lr.pkl"

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Car Selling Price Predictor")
st.caption("Linear Regression trained on the bundled car dataset.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

with st.sidebar:
    st.header("Input car details")
    year = st.number_input("Year", 1995, 2025, 2015, 1)
    present_price = st.number_input("Present Price (lakhs)", 0.0, 100.0, 6.0, 0.1)
    kms_driven = st.number_input("Kms Driven", 0, 500000, 30000, 500)
    owner = st.selectbox("Previous owners", [0, 1, 2, 3], index=0)
    fuel_type = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].unique()))
    seller_type = st.selectbox("Seller Type", sorted(df["Seller_Type"].unique()))
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
    predict = st.button("Predict selling price", type="primary", use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    if predict:
        X = pd.DataFrame([{
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Age": 2020 - year,
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
        }])
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)
        st.metric("Estimated selling price", f"₹ {pred:.2f} lakhs")
        st.write("Input used:")
        st.dataframe(X, use_container_width=True)
    else:
        st.info("Fill in the sidebar and click **Predict selling price**.")

with col2:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True, height=380)

with st.expander("Dataset statistics"):
    st.write(f"Rows: {len(df)}")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
st.header("📊 Data Visualization")

# Graph 1: Selling Price Distribution
fig1 = px.histogram(
    df,
    x="Selling_Price",
    nbins=30,
    title="Distribution of Selling Price"
)
st.plotly_chart(fig1, use_container_width=True)

# Graph 2: Fuel Type Count
fig2 = px.pie(
    df,
    names="Fuel_Type",
    title="Fuel Type Distribution"
)
st.plotly_chart(fig2, use_container_width=True)

# Graph 3: Transmission Count
fig3 = px.bar(
    df["Transmission"].value_counts().reset_index(),
    x="Transmission",
    y="count",
    title="Transmission Types"
)
st.plotly_chart(fig3, use_container_width=True)

# Graph 4: Present Price vs Selling Price
fig4 = px.scatter(
    df,
    x="Present_Price",
    y="Selling_Price",
    color="Fuel_Type",
    title="Present Price vs Selling Price"
)
st.plotly_chart(fig4, use_container_width=True)

# Graph 5: Year vs Selling Price
fig5 = px.line(
    df.sort_values("Year"),
    x="Year",
    y="Selling_Price",
    title="Selling Price by Year"
)
st.plotly_chart(fig5, use_container_width=True)
