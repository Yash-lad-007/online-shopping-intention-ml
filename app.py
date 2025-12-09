# app/streamlit_app.py

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils import load_trained_model, make_prediction

st.set_page_config(page_title="Online Shopping Intention Analysis", layout="centered")

@st.cache_resource
def get_model():
    return load_trained_model()

model = get_model()

st.title("🛒 Online Shopping Intention Analysis")
st.write(
    "Predict whether an online shopper is likely to **complete a purchase** "
    "based on their session behavior."
)

st.sidebar.header("Session Features")

# Example inputs (match your dataset!)
administrative = st.sidebar.number_input("Administrative pages", min_value=0, max_value=30, value=2)
administrative_duration = st.sidebar.number_input("Administrative Duration", min_value=0.0, value=20.0)
informational = st.sidebar.number_input("Informational pages", min_value=0, max_value=30, value=1)
informational_duration = st.sidebar.number_input("Informational Duration", min_value=0.0, value=10.0)
product_related = st.sidebar.number_input("Product Related pages", min_value=0, max_value=500, value=30)
product_related_duration = st.sidebar.number_input("Product Related Duration", min_value=0.0, value=200.0)
bounce_rates = st.sidebar.number_input("Bounce Rates", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f")
exit_rates = st.sidebar.number_input("Exit Rates", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f")
page_values = st.sidebar.number_input("Page Values", min_value=0.0, value=10.0)
special_day = st.sidebar.number_input("Special Day proximity", min_value=0.0, max_value=1.0, value=0.0)

month = st.sidebar.selectbox("Month", [
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])

operating_systems = st.sidebar.selectbox("Operating System", [1, 2, 3, 4, 5, 6, 7, 8])
browser = st.sidebar.selectbox("Browser", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
region = st.sidebar.selectbox("Region", [1, 2, 3, 4, 5, 6, 7, 8, 9])
traffic_type = st.sidebar.selectbox("Traffic Type", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

visitor_type = st.sidebar.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"])
weekend = st.sidebar.selectbox("Weekend?", [False, True])

input_data = {
    "Administrative": administrative,
    "Administrative_Duration": administrative_duration,
    "Informational": informational,
    "Informational_Duration": informational_duration,
    "ProductRelated": product_related,
    "ProductRelated_Duration": product_related_duration,
    "BounceRates": bounce_rates,
    "ExitRates": exit_rates,
    "PageValues": page_values,
    "SpecialDay": special_day,
    "Month": month,
    "OperatingSystems": operating_systems,
    "Browser": browser,
    "Region": region,
    "TrafficType": traffic_type,
    "VisitorType": visitor_type,
    "Weekend": weekend,
}

st.subheader("Input Summary")
st.write(pd.DataFrame([input_data]).T.rename(columns={0: "Value"}))

if st.button("Predict Purchase Intention"):
    pred, proba = make_prediction(model, input_data)
    if pred == 1:
        st.success(f"✅ The model predicts the user is **likely to buy** (probability: {proba:.2%})")
    else:
        st.warning(f"❌ The model predicts the user is **unlikely to buy** (probability of purchase: {proba:.2%})")

    st.caption("This is a machine learning prediction, not a guarantee.")
