import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing object
model = joblib.load("model.joblib")
pre = joblib.load("pre.joblib")

st.title("Walmart Retail Demand Forecasting")

st.write("Predict Weekly Sales using Machine Learning")

st.write("---")

# User Inputs

store = st.number_input("Store", min_value=1, max_value=50, value=1)

holiday_flag = st.selectbox("Holiday Flag", [0,1])

temperature = st.number_input("Temperature")

fuel_price = st.number_input("Fuel Price")

cpi = st.number_input("CPI")

unemployment = st.number_input("Unemployment")

year = st.number_input("Year", min_value=2010, max_value=2030)

month = st.slider("Month",1,12)

week = st.slider("Week",1,52)


# Create dataframe from inputs

input_df = pd.DataFrame({
    "Store":[store],
    "Holiday_Flag":[holiday_flag],
    "Temperature":[temperature],
    "Fuel_Price":[fuel_price],
    "CPI":[cpi],
    "Unemployment":[unemployment],
    "year":[year],
    "month":[month],
    "week":[week]
})


st.write("### Input Data")
st.dataframe(input_df)


# Prediction

if st.button("Predict Weekly Sales"):

    input_processed = pre.transform(input_df)

    prediction = model.predict(input_processed)

    st.success(f"Predicted Weekly Sales: {prediction[0]:,.2f}")