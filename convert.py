import pandas as pd
import numpy as np

# 1. Load datasets
price_df = pd.read_csv('crop_price_data.csv')
prod_df = pd.read_csv('crop_production_data.csv')
soil_df = pd.read_csv('soil_analysis_data.csv')
water_df = pd.read_csv('water_usage_data.csv')

# 2. Mapping Logic (India -> Bangladesh)
district_map = {
    'Jaipur': 'Dhaka', 'Kota': 'Chittagong', 'Alwar': 'Cumilla', 
    'Nagaur': 'Bogra', 'Sri Ganganagar': 'Rangpur', 'Jodhpur': 'Jessore', 
    'Udaipur': 'Sylhet', 'Bhilwara': 'Rajshahi', 'Ajmer': 'Barisal', 
    'Hanumangarh': 'Mymensingh', 'Tonk': 'Khulna', 'Baran': 'Dinajpur'
}

bd_districts = list(district_map.values())

def get_bd_district(indian_dist):
    return district_map.get(indian_dist, np.random.choice(bd_districts))

# 3. Process & Rename (File by File)

# --- Crop Price Data ---
price_df['District'] = price_df['District'].apply(get_bd_district)
price_df['Market'] = price_df['District'] + " Sadar Bazar"
price_df['Price (INR/quintal)'] = (price_df['Price (INR/quintal)'] / 100) * 1.6  # Convert to Tk/kg
price_df['Price (INR/quintal)'] = price_df['Price (INR/quintal)'].round(2)
price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce') + pd.DateOffset(years=6) # 2018 -> 2024
price_df = price_df.rename(columns={
    'District': 'District_Name', 'Crop': 'Crop_Name', 'Market': 'Market_Name', 
    'Date': 'Price_Date', 'Price (INR/quintal)': 'Price_Tk_kg'
})

# --- Crop Production Data ---
prod_df['District'] = prod_df['District'].apply(get_bd_district)
prod_df = prod_df.rename(columns={
    'District': 'District_Name', 'Crop': 'Crop_Name', 'Season': 'Season', 
    'Area (hectares)': 'Cultivated_Area_Hectares', 
    'Yield (quintals)': 'Yield_Quintals_per_Ha', 
    'Production (metric tons)': 'Production_Metric_Tons'
})

# --- Soil Analysis Data ---
soil_df['District'] = soil_df['District'].apply(get_bd_district)
soil_df = soil_df.rename(columns={
    'District': 'District_Name', 'Soil Type': 'Soil_Type', 'pH Level': 'pH_Level',
    'Organic Matter (%)': 'Organic_Matter_Percent', 
    'Nitrogen Content (kg/ha)': 'Nitrogen_Content_kg_ha', 
    'Phosphorus Content (kg/ha)': 'Phosphorus_Content_kg_ha', 
    'Potassium Content (kg/ha)': 'Potassium_Content_kg_ha'
})

# --- Water Usage Data ---
water_df['District'] = water_df['District'].apply(get_bd_district)
water_df = water_df.rename(columns={
    'District': 'District_Name', 'Crop': 'Crop_Name', 'Irrigation Method': 'Irrigation_Method',
    'Water Consumption (liters/hectare)': 'Water_Consumption_L_ha',
    'Water Availability (liters/hectare)': 'Water_Availability_L_ha'
})

# 4. Save New Files
price_df.to_csv('bd_crop_price_data.csv', index=False)
prod_df.to_csv('bd_crop_production_data.csv', index=False)
soil_df.to_csv('bd_soil_analysis_data.csv', index=False)
water_df.to_csv('bd_water_usage_data.csv', index=False)

print("All files converted successfully!")