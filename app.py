import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# 1. Define Model Architecture
class VehicleMLP(nn.Module):
    def __init__(self, input_dim):
        super(VehicleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2. Load Pipeline and Weights
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor_pipeline.joblib")
    input_dim = len(preprocessor.get_feature_names_out())
    model = VehicleMLP(input_dim)
    model.load_state_dict(torch.load("vehicle_model_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    return preprocessor, model

preprocessor, model = load_artifacts()

# 3. User Interface
st.set_page_config(page_title="Used Car Valuation & Depreciation Assistant", layout="wide")
st.title("Used Car Valuation & Depreciation Assistant")
st.markdown("Enter the vehicle specifications below to calculate an instant fair-market valuation and projected 5-year depreciation trajectory.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Specifications")
    brand = st.selectbox("Brand / Make", ['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Ford', 'Honda', 'Toyota', 'Chevrolet', 'Renault', 'Other'])
    year = st.slider("Year of Manufacture", 2000, 2026, 2017)
    km_driven = st.number_input("Odometer Mileage (km)", min_value=500, max_value=400000, value=65000, step=1000)
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    owner = st.selectbox("Previous Owners", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

CURRENT_YEAR = 2026
vehicle_age = CURRENT_YEAR - year
km_per_year = km_driven / (vehicle_age + 1)

input_dict = {
    'brand': brand,
    'fuel': fuel,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner,
    'vehicle_age': vehicle_age,
    'km_driven': km_driven,
    'km_per_year': km_per_year
}

with col2:
    st.subheader("Market Valuation & Trajectory")
    
    if st.button("Calculate Valuation & Trajectory", type="primary"):
        # Preprocess input
        input_df = pd.DataFrame([input_dict])
        input_tensor = torch.tensor(preprocessor.transform(input_df), dtype=torch.float32)
        
        with torch.no_grad():
            curr_log_val = model(input_tensor).item()
            curr_val = np.exp(curr_log_val)
            
        # Display Metrics
        st.metric("Estimated Fair Market Value", f"KES{curr_val:,.2f}")
        st.write(f"Fair Deal Range: **KES{curr_val*0.92:,.2f}** – **KES{curr_val*1.08:,.2f}**")
        
        # 5-Year Projection Simulation
        years = list(range(0, 6))
        proj_vals = []
        for t in years:
            sim = input_dict.copy()
            sim['vehicle_age'] += t
            sim['km_driven'] += (t * 15000)
            sim['km_per_year'] = sim['km_driven'] / (sim['vehicle_age'] + 1)
            t_df = pd.DataFrame([sim])
            t_tensor = torch.tensor(preprocessor.transform(t_df), dtype=torch.float32)
            with torch.no_grad():
                proj_vals.append(np.exp(model(t_tensor).item()))
                
        # Trajectory Chart
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot([CURRENT_YEAR + t for t in years], proj_vals, marker='o', color='#1E88E5', linewidth=2)
        ax.set_ylabel("Predicted Valuation (KES)")
        ax.set_xlabel("Year")
        ax.set_title("5-Year Depreciation Trajectory (15,000 km/yr)")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)