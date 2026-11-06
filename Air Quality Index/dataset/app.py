
# --------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# --------------------------------------------------

import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# STEP 2: PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Air Quality Index Prediction",
    page_icon="🌍",
    layout="wide"
)


# --------------------------------------------------
# STEP 3: LOAD THE TRAINED MODEL
# --------------------------------------------------

model = joblib.load("model.joblib")

feature_names = joblib.load("feature_names.joblib")

feature_medians = joblib.load("feature_medians.joblib")


# --------------------------------------------------
# STEP 4: DISPLAY APPLICATION TITLE
# --------------------------------------------------

st.title("🌍 Air Quality Index Prediction Application")

st.write(
    "Predict Air Quality Index using environmental data."
)


# --------------------------------------------------
# STEP 5: CREATE THE SIDEBAR
# --------------------------------------------------

st.sidebar.header("Environmental Information")


# --------------------------------------------------
# STEP 6: COLLECT ENVIRONMENTAL INPUTS
# --------------------------------------------------

# The input fields are generated from the dataset's
# actual feature names and their median values.

input_values = {}

for feature in feature_names:

    default_value = float(feature_medians[feature])

    input_values[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(),
        value=default_value,
        format="%.2f",
        key=feature
    )


# --------------------------------------------------
# STEP 7: CREATE A DATAFRAME FROM USER INPUTS
# --------------------------------------------------

input_df = pd.DataFrame(
    [input_values],
    columns=feature_names
)


# --------------------------------------------------
# STEP 8: DISPLAY ENVIRONMENTAL INFORMATION
# --------------------------------------------------

st.subheader("Environmental Data Details")

st.dataframe(
    input_df,
    use_container_width=True
)


# --------------------------------------------------
# STEP 9: PREDICT AIR QUALITY INDEX
# --------------------------------------------------

if st.button("Predict AQI", type="primary"):

    # Predict the numerical AQI value
    prediction = float(model.predict(input_df)[0])

    # AQI cannot be negative
    prediction = max(0.0, prediction)

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted PM2.5 AQI",
        value=f"{prediction:.2f}"
    )

    # Illustrative category classification using
    # commonly used US EPA AQI breakpoints.
    if prediction <= 50:
        category = "Good"

    elif prediction <= 100:
        category = "Moderate"

    elif prediction <= 150:
        category = "Unhealthy for Sensitive Groups"

    elif prediction <= 200:
        category = "Unhealthy"

    elif prediction <= 300:
        category = "Very Unhealthy"

    else:
        category = "Hazardous"

    st.write("Air Quality Category:", category)

    st.info(
        "This is a machine learning estimate based on "
        "environmental measurements. It is not a live AQI reading."
    )
