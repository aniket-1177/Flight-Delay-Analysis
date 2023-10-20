import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel

# Initialize Spark session
spark = SparkSession.builder.appName("SparkModelApp").getOrCreate()

# Load the saved PySpark model
gbt_model = GBTClassificationModel.load("gbt_model")

# Create a function to make predictions
def predict_delay(depart, duration, km, org_idx, carrier_idx):
    prediction = gbt_model.transform(spark.createDataFrame([(depart, duration, km, org_idx, carrier_idx)],
                                                          ["depart", "duration", "km", "org_idx", "carrier_idx"]))
    return prediction.select("probability").collect()[0][0][1]

# Streamlit UI
st.title("Flight Delay Prediction")
st.sidebar.title("User Input")

depart = st.sidebar.number_input("Departure Time", min_value=0, max_value=2400)
duration = st.sidebar.number_input("Duration (minutes)", min_value=0, max_value=1000)
km = st.sidebar.number_input("Distance (km)", min_value=0, max_value=10000)
org_idx = st.sidebar.number_input("Origin Index", min_value=0)
carrier_idx = st.sidebar.number_input("Carrier Index", min_value=0)

if st.sidebar.button("Predict Delay"):
    prediction = predict_delay(depart, duration, km, org_idx, carrier_idx)
    if prediction > 0.5:
        st.error(f"The flight is likely to be delayed (Probability: {prediction:.2f})")
    else:
        st.success(f"The flight is likely to be on time (Probability: {1 - prediction:.2f})")
