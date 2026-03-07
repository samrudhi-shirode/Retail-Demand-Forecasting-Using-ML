import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")

st.title("Walmart Retail Demand Forecasting")

store = st.number_input("Store Number")
holiday_flag = st.selectbox("Holiday Flag",[0,1])
temperature = st.number_input("Temperature")
fuel_price = st.number_input("Fuel Price")
cpi = st.number_input("CPI")
unemployment = st.number_input("Unemployment")
year = st.number_input("Year")
month = st.slider("Month",1,12)
week = st.slider("Week",1,52)

input_data = pd.DataFrame({
    'Store':[store],
    'Holiday_Flag':[holiday_flag],
    'Temperature':[temperature],
    'Fuel_Price':[fuel_price],
    'CPI':[cpi],
    'Unemployment':[unemployment],
    'year':[year],
    'month':[month],
    'week':[week]
})

if st.button("Predict Sales"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Weekly Sales: {prediction[0]:,.2f}")