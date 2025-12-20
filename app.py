# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib 
from pathlib import Path

MODEL_PATH = "models/best_model.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


model = load_model()

st.set_page_config(
    page_title="Online Shopping Intention Analysis",
    layout="centered",
)

st.title("🛒 Online Shopping Intention Analysis")
st.write(
    "This app predicts whether an online shopper is likely to **complete a purchase** "
    "based on their session behavior."
)

st.sidebar.header("Enter Session Details")

# --- Inputs (match your dataset columns) ---
administrative = st.sidebar.number_input("Administrative pages", min_value=0, max_value=30, value=2)
administrative_duration = st.sidebar.number_input("Administrative Duration", min_value=0.0, value=20.0)
informational = st.sidebar.number_input("Informational pages", min_value=0, max_value=30, value=1)
informational_duration = st.sidebar.number_input("Informational Duration", min_value=0.0, value=10.0)
product_related = st.sidebar.number_input("Product Related pages", min_value=0, max_value=500, value=30)
product_related_duration = st.sidebar.number_input("Product Related Duration", min_value=0.0, value=200.0)
bounce_rates = st.sidebar.number_input("Bounce Rates", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f")
exit_rates = st.sidebar.number_input("Exit Rates", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f")
page_values = st.sidebar.number_input("Page Values", min_value=0.0, value=10.0)
special_day = st.sidebar.number_input("Special Day (0–1, proximity to a special day)", min_value=0.0, max_value=1.0, value=0.0)

month = st.sidebar.selectbox(
    "Month",
    ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
)

operating_systems = st.sidebar.selectbox("Operating System", [1, 2, 3, 4, 5, 6, 7, 8])
browser = st.sidebar.selectbox("Browser", list(range(1, 14)))
region = st.sidebar.selectbox("Region", list(range(1, 10)))
traffic_type = st.sidebar.selectbox("Traffic Type", list(range(1, 21)))

visitor_type = st.sidebar.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"])
weekend = st.sidebar.selectbox("Weekend?", [False, True])

# Build row as dataframe
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

st.subheader("Current Input")
st.write(pd.DataFrame([input_data]))

if st.button("Predict Purchase Intention"):
    X_input = pd.DataFrame([input_data])
    proba = model.predict_proba(X_input)[0][1]
    pred = int(proba >= 0.5)

    if pred == 1:
        st.success(f"✅ The user is **likely to make a purchase**.\n\nPurchase probability: **{proba:.2%}**")
    else:
        st.warning(f"❌ The user is **unlikely to make a purchase**.\n\nPurchase probability: **{proba:.2%}**")

    st.caption("Note: This is a machine learning prediction, not a guarantee.")
