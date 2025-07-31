import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- Load Model and Data ---
@st.cache_resource
def load_model_and_data():
    """
    Loads the pre-trained CatBoost model and the original dataset.
    """
    try:
        model = joblib.load('catboost_model.pkl')
        # We still load df_original to get the unique values for selectboxes
        df_original = pd.read_csv('used_car_price.csv')
        return model, df_original
    except FileNotFoundError:
        st.error("Error: 'catboost_model.pkl' or 'used_car_price.csv' not found.")
        st.error("Please ensure both files are in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.stop()

model, df_original = load_model_and_data()

# --- Define Categorical Columns Info ---
# Ensure 'service_history' has 'None' for consistency in options
df_original['service_history'].fillna("None", inplace=True)

categorical_cols_info = {
    'fuel_type': sorted(df_original['fuel_type'].unique().tolist()),
    'brand': sorted(df_original['brand'].unique().tolist()),
    'transmission': sorted(df_original['transmission'].unique().tolist()),
    'color': sorted(df_original['color'].unique().tolist()),
    'service_history': sorted(df_original['service_history'].unique().tolist()),
    'insurance_valid': sorted(df_original['insurance_valid'].unique().tolist())
}

# --- Streamlit App UI ---
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 15px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #e6f7ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin-top: 30px;
        border-radius: 5px;
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details below to get a price prediction.")

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    make_year = st.number_input("Make Year", min_value=1995, max_value=2023, value=2015, step=1,
                                help="Year the car was manufactured.")
    mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=35.0, value=18.0, step=0.1, format="%.2f",
                                   help="Fuel efficiency in kilometers per liter.")
    engine_cc = st.number_input("Engine CC", min_value=800, max_value=5000, value=1500, step=100,
                                help="Engine displacement in cubic centimeters.")
    owner_count = st.selectbox("Owner Count", options=[1, 2, 3, 4, 5], index=0,
                               help="Number of previous owners.")
    accidents_reported = st.number_input("Accidents Reported", min_value=0, max_value=5, value=0, step=1,
                                        help="Number of accidents reported for the car.")

with col2:
    fuel_type = st.selectbox("Fuel Type", options=categorical_cols_info['fuel_type'], index=0,
                             help="Type of fuel the car uses.")
    brand = st.selectbox("Brand", options=categorical_cols_info['brand'], index=0,
                         help="Manufacturer of the car.")
    transmission = st.selectbox("Transmission", options=categorical_cols_info['transmission'], index=0,
                                help="Type of transmission (Manual or Automatic).")
    color = st.selectbox("Color", options=categorical_cols_info['color'], index=0,
                         help="Color of the car.")
    service_history = st.selectbox("Service History", options=categorical_cols_info['service_history'], index=categorical_cols_info['service_history'].index('None'), # Default to 'None'
                                   help="Availability of service history (Full, Partial, or None).")
    insurance_valid = st.selectbox("Insurance Valid", options=categorical_cols_info['insurance_valid'], index=categorical_cols_info['insurance_valid'].index('Yes'), # Default to 'Yes'
                                   help="Is the car's insurance valid? (Yes/No).")

# --- Prediction Button ---
if st.button("Predict Price"):
    # 1. Feature Engineering
    car_age = 2025 - make_year

    # 2. Create DataFrame for Prediction with original string values for categorical features
    # Ensure the order of columns matches the training data
    input_data = pd.DataFrame([[
        car_age,
        mileage_kmpl,
        engine_cc,
        fuel_type, # Pass original string
        owner_count,
        brand,       # Pass original string
        transmission, # Pass original string
        color,       # Pass original string
        service_history, # Pass original string
        accidents_reported,
        insurance_valid # Pass original string
    ]], columns=[
        'car_age', 'mileage_kmpl', 'engine_cc', 'fuel_type', 'owner_count',
        'brand', 'transmission', 'color', 'service_history',
        'accidents_reported', 'insurance_valid'
    ])

    # 3. Make Prediction
    try:
        predicted_price = model.predict(input_data)[0]
        st.markdown(f"<div class='prediction-box'>Predicted Price: ${predicted_price:.2f} USD</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values.")

