import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agri-Smart BD | এআই মূল্য পূর্বাভাস",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a stunning, professional dashboard design
st.markdown("""
    <style>
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content area styling */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-top: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* All text elements */
    p, span, div, label, .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #1a1a1a !important;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border-left: 5px solid #28a745 !important;
    }
    
    .stSuccess > div, .stInfo > div, .stWarning > div {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stInfo {
        border-left-color: #17a2b8 !important;
    }
    
    .stWarning {
        border-left-color: #ffc107 !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox label, .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Selectbox dropdown styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
    }
    
    /* Selectbox selected value */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
    }
    
    /* Dropdown menu options list */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    /* Individual dropdown options */
    [role="option"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Dropdown option on hover */
    [role="option"]:hover {
        background-color: #667eea !important;
        color: #ffffff !important;
    }
    
    /* Selected option in dropdown */
    [aria-selected="true"] {
        background-color: #764ba2 !important;
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0,0,0,0.1) !important;
    }
    
    /* Cards effect for metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
    
    /* Footer styling */
    footer {
        color: #1a1a1a !important;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Loads the pre-processed Bangladesh agricultural datasets.
    Uses caching to speed up performance.
    """
    try:
        # Load CSV files
        price_df = pd.read_csv('bd_crop_price_data.csv')
        prod_df = pd.read_csv('bd_crop_production_data.csv')
        soil_df = pd.read_csv('bd_soil_analysis_data.csv')
        
        # Ensure Date column is in datetime format
        price_df['Price_Date'] = pd.to_datetime(price_df['Price_Date'])
        
        return price_df, prod_df, soil_df
    except FileNotFoundError as e:
        return None, None, None

# Load the data
price_df, prod_df, soil_df = load_data()

# Error handling if data is missing
if price_df is None:
    st.error("🚨 ত্রুটি: ডেটাসেট ফাইল পাওয়া যায়নি! অনুগ্রহ করে নিশ্চিত করুন যে 'bd_crop_price_data.csv' ফোল্ডারে আছে।")
    st.stop()

# -----------------------------------------------------------------------------
# TRANSLATION DICTIONARIES
# -----------------------------------------------------------------------------
# Translate crop names to Bengali
crop_translation = {
    'Rice': 'ধান',
    'Wheat': 'গম',
    'Jute': 'পাট',
    'Potato': 'আলু',
    'Onion': 'পেঁয়াজ',
    'Garlic': 'রসুন',
    'Lentil': 'ডাল',
    'Mustard': 'সরিষা',
    'Tomato': 'টমেটো',
    'Eggplant': 'বেগুন',
    'Cabbage': 'বাঁধাকপি',
    'Cauliflower': 'ফুলকপি',
    'Chili': 'মরিচ',
    'Cucumber': 'শসা',
    'Pumpkin': 'কুমড়া',
    'Bitter Gourd': 'করলা',
    'Bottle Gourd': 'লাউ',
    'Okra': 'ঢেঁড়স',
    'Spinach': 'পালং শাক',
    'Coriander': 'ধনিয়া',
    'Maize': 'ভুট্টা',
    'Sugarcane': 'আখ',
    'Tea': 'চা',
    'Mango': 'আম',
    'Banana': 'কলা',
    'Jackfruit': 'কাঁঠাল',
    'Papaya': 'পেঁপে',
    'Guava': 'পেয়ারা',
    'Lychee': 'লিচু',
    'Pineapple': 'আনারস',
    'Bajra': 'বাজরা',
    'Barley': 'যব',              
    'Chilli': 'মরিচ',
    'Citrus': 'লেবুজাতীয় ফল',    
    'Cotton': 'তুলা',         
    'Cumin': 'জিরা',
    'Fennel': 'মৌরি',         
    'Fenugreek': 'মেথি',
    'Gram': 'ছোলা',           
    'Oilseeds': 'তেলবীজ',
    'Opium': 'আফিম',         
    'Pomegranate': 'ডালিম',    
    'Pulses': 'ডালশস্য' 
    
}

# Translate soil types to Bengali
soil_translation = {
    'Clay': 'কর্দম মাটি',
    'Loamy': 'দোআঁশ মাটি',
    'Sandy': 'বেলে মাটি',
    'Silt': 'পলি মাটি',
    'Clay Loam': 'কর্দম দোআঁশ',
    'Sandy Loam': 'বেলে দোআঁশ',
    'Silty Clay': 'পলি কর্দম',
    'Silty Loam': 'পলি দোআঁশ',
    'Peat': 'পিট মাটি',
    'Chalky (Calcareous)': 'চুনযুক্ত মাটি',
    'Nitrogenous': 'নাইট্রোজেন সমৃদ্ধ',
    'Black lava soil': 'কালো লাভা মাটি',
}

# District translation dictionary
district_translation = {
    'Dhaka': 'ঢাকা',
    'Chittagong': 'চট্টগ্রাম',
    'Rajshahi': 'রাজশাহী',
    'Khulna': 'খুলনা',
    'Barisal': 'বরিশাল',
    'Sylhet': 'সিলেট',
    'Rangpur': 'রংপুর',
    'Mymensingh': 'ময়মনসিংহ',
    'Comilla': 'কুমিল্লা',
    'Gazipur': 'গাজীপুর',
    'Narayanganj': 'নারায়ণগঞ্জ',
    'Tangail': 'টাঙ্গাইল',
    'Jamalpur': 'জামালপুর',
    'Bogra': 'বগুড়া',
    'Pabna': 'পাবনা',
    'Jessore': 'যশোর',
    'Dinajpur': 'দিনাজপুর',
    'Faridpur': 'ফরিদপুর',
    'Kushtia': 'কুষ্টিয়া',
    'Noakhali': 'নোয়াখালী',
    'Brahmanbaria': 'ব্রাহ্মণবাড়িয়া',
    'Feni': 'ফেনী',
    'Lakshmipur': 'লক্ষ্মীপুর',
    'Chandpur': 'চাঁদপুর',
    'Kishoreganj': 'কিশোরগঞ্জ',
    'Netrokona': 'নেত্রকোনা',
    'Sherpur': 'শেরপুর',
    'Habiganj': 'হবিগঞ্জ',
    'Moulvibazar': 'মৌলভীবাজার',
    'Sunamganj': 'সুনামগঞ্জ',
    'Narsingdi': 'নরসিংদী',
    'Munshiganj': 'মুন্সিগঞ্জ',
    'Manikganj': 'মানিকগঞ্জ',
    'Gopalganj': 'গোপালগঞ্জ',
    'Madaripur': 'মাদারীপুর',
    'Shariatpur': 'শরীয়তপুর',
    'Rajbari': 'রাজবাড়ী',
    'Magura': 'মাগুরা',
    'Jhenaidah': 'ঝিনাইদহ',
    'Narail': 'নড়াইল',
    'Satkhira': 'সাতক্ষীরা',
    'Bagerhat': 'বাগেরহাট',
    'Pirojpur': 'পিরোজপুর',
    'Jhalokati': 'ঝালকাঠি',
    'Patuakhali': 'পটুয়াখালী',
    'Barguna': 'বরগুনা',
    'Sirajganj': 'সিরাজগঞ্জ',
    'Natore': 'নাটোর',
    'Chapainawabganj': 'চাঁপাইনবাবগঞ্জ',
    'Naogaon': 'নওগাঁ',
    'Joypurhat': 'জয়পুরহাট',
    'Gaibandha': 'গাইবান্ধা',
    'Kurigram': 'কুড়িগ্রাম',
    'Lalmonirhat': 'লালমনিরহাট',
    'Nilphamari': 'নীলফামারী',
    'Panchagarh': 'পঞ্চগড়',
    'Thakurgaon': 'ঠাকুরগাঁও',
    'Coxs Bazar': 'কক্সবাজার',
    'Bandarban': 'বান্দরবান',
    'Rangamati': 'রাঙ্গামাটি',
    'Khagrachari': 'খাগড়াছড়ি',
    'Meherpur': 'মেহেরপুর',
    'Chuadanga': 'চুয়াডাঙ্গা',
    'Cumilla': 'কুমিল্লা'
}

# Helper function to translate text
def translate_bn(text, translation_dict):
    return translation_dict.get(text, text)

# Convert English numbers to Bengali numbers
def to_bengali_number(number):
    """Convert English numerals to Bengali numerals"""
    bengali_digits = {'0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪', 
                      '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯', '.': '.'}
    return ''.join(bengali_digits.get(char, char) for char in str(number))

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🌾 Agri-Smart BD")
st.sidebar.markdown("**এআই চালিত কৃষি বুদ্ধিমত্তা**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "মডিউল নির্বাচন করুন:",
    ["📊 মূল্য পূর্বাভাস (এআই)", "💰 সেরা বাজার খুঁজুন", "🌱 মাটি ও ফসল পরামর্শদাতা"]
)

st.sidebar.markdown("---")
st.sidebar.info(
     "**Project:** Farm-to-Market Intelligence\n"
    "**Team:** Million Minds\n"
    "**Event:** AI Build-a-thon 2025"
)

# -----------------------------------------------------------------------------
# 4. MODULE 1: AI PRICE FORECASTING
# -----------------------------------------------------------------------------
if menu == "📊 মূল্য পূর্বাভাস (এআই)":
    st.title("📊 এআই চালিত মূল্য পূর্বাভাস")
    st.markdown("<h3 style='color: #1a1a1a; font-weight: 500;'>মেশিন লার্নিং (র‍্যান্ডম ফরেস্ট) ব্যবহার করে ভবিষ্যতের ফসলের দাম পূর্বাভাস করুন যাতে কৃষকরা ভালো বিক্রয় সিদ্ধান্ত নিতে পারেন।</h3>", unsafe_allow_html=True)
    st.divider()

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        # Select District
        district_list = sorted(price_df['District_Name'].unique())
        district_display = {dist: translate_bn(dist, district_translation) for dist in district_list}
        selected_district_bn = st.selectbox("📍 জেলা নির্বাচন করুন", 
                                             options=list(district_display.values()),
                                             format_func=lambda x: x)
        selected_district = [k for k, v in district_display.items() if v == selected_district_bn][0]
    
    with col2:
        # Select Crop (Filtered by District)
        available_crops = sorted(price_df[price_df['District_Name'] == selected_district]['Crop_Name'].unique())
        # Create Bengali display names
        crop_display = {crop: translate_bn(crop, crop_translation) for crop in available_crops}
        selected_crop_bn = st.selectbox("🌽 ফসল নির্বাচন করুন", 
                                         options=list(crop_display.values()),
                                         format_func=lambda x: x)
        # Get English name for data filtering
        selected_crop = [k for k, v in crop_display.items() if v == selected_crop_bn][0]

    # --- Data Processing ---
    # Filter data for specific district and crop
    filtered_df = price_df[
        (price_df['District_Name'] == selected_district) & 
        (price_df['Crop_Name'] == selected_crop)
    ].sort_values('Price_Date')

    if len(filtered_df) > 10:
        # --- MACHINE LEARNING SECTION (IMPROVED) ---
        
        # 1. Feature Engineering: Add Seasonality
        filtered_df['Month'] = filtered_df['Price_Date'].dt.month
        filtered_df['Week'] = filtered_df['Price_Date'].dt.isocalendar().week
        filtered_df['Year'] = filtered_df['Price_Date'].dt.year
        filtered_df['Date_Ordinal'] = filtered_df['Price_Date'].map(datetime.datetime.toordinal)
        
        # 2. Define Features (X) and Target (y)
        # We use Month and Week to capture seasonal trends
        X = filtered_df[['Date_Ordinal', 'Month', 'Week', 'Year']]
        y = filtered_df['Price_Tk_kg']
        
        # 3. Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 4. Generate Future Data
        last_date = filtered_df['Price_Date'].max()
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
        
        # Create features for future dates
        future_data = pd.DataFrame({
            'Price_Date': future_dates
        })
        future_data['Date_Ordinal'] = future_data['Price_Date'].map(datetime.datetime.toordinal)
        future_data['Month'] = future_data['Price_Date'].dt.month
        future_data['Week'] = future_data['Price_Date'].dt.isocalendar().week
        future_data['Year'] = future_data['Price_Date'].dt.year
        
        # 5. Predict
        future_prices = model.predict(future_data[['Date_Ordinal', 'Month', 'Week', 'Year']])
        
        # 6. Prepare Data for Visualization
        future_df = pd.DataFrame({
            'Price_Date': future_dates,
            'Price_Tk_kg': future_prices,
            'Type': 'এআই পূর্বাভাস'
        })
        
        filtered_df['Type'] = 'ঐতিহাসিক তথ্য'
        combined_df = pd.concat([filtered_df[['Price_Date', 'Price_Tk_kg', 'Type']], future_df])

        # --- VISUALIZATION ---
        st.subheader(f"মূল্য প্রবণতা বিশ্লেষণ: {translate_bn(selected_crop, crop_translation)}")
        
        # Interactive Line Chart using Plotly
        fig = px.line(
            combined_df, 
            x='Price_Date', 
            y='Price_Tk_kg', 
            color='Type',
            color_discrete_map={
                "ঐতিহাসিক তথ্য": "#1f77b4", # Blue
                "এআই পূর্বাভাস": "#00cc96" # Green
            },
            title=f"{translate_bn(selected_district, district_translation)} এ {translate_bn(selected_crop, crop_translation)} এর {to_bengali_number(30)} দিনের মূল্য পূর্বাভাস",
            labels={'Price_Tk_kg': 'মূল্য (টাকা / কেজি)', 'Price_Date': 'তারিখ'}
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- INSIGHTS & METRICS ---
        current_price = filtered_df.iloc[-1]['Price_Tk_kg']
        avg_forecast_price = future_prices.mean()
        
        # Determine Trend Logic
        if avg_forecast_price > current_price:
            trend_label = "বৃদ্ধির প্রবণতা 📈"
            trend_color = "normal"
        else:
            trend_label = "হ্রাসের প্রবণতা 📉"
            trend_color = "inverse"

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("বর্তমান বাজার মূল্য", f"৳ {to_bengali_number(f'{current_price:.2f}')} প্রতি কেজি")
        m2.metric(f"পূর্বাভাসিত গড় (পরবর্তী {to_bengali_number(30)} দিন)", f"৳ {to_bengali_number(f'{avg_forecast_price:.2f}')} প্রতি কেজি")
        m3.metric("বাজার অবস্থা", trend_label, delta_color=trend_color)

        # Actionable Advice
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2); margin-top: 1rem;'>
            <h3 style='color: white !important; margin: 0;'>💡 এআই সুপারিশ</h3>
            <p style='color: white !important; font-size: 1.1rem; margin-top: 0.5rem;'>
                পূর্বাভাসের ভিত্তিতে, আপনি যদি পরবর্তী সপ্তাহে <b>{translate_bn(selected_crop, crop_translation)}</b> বিক্রি করেন, 
                তাহলে আপনি গড়ে <b style='font-size: 1.3rem;'>৳{to_bengali_number(f'{future_prices[:7].mean():.2f}')}</b> মূল্য পেতে পারেন
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ এই নির্বাচনের জন্য সঠিক পূর্বাভাস তৈরি করার জন্য পর্যাপ্ত ঐতিহাসিক তথ্য নেই।")

# -----------------------------------------------------------------------------
# 5. MODULE 2: BEST MARKET FINDER
# -----------------------------------------------------------------------------
elif menu == "💰 সেরা বাজার খুঁজুন":
    st.title("💰 স্মার্ট বাজার খুঁজুন")
    st.markdown("<h3 style='color: #1a1a1a; font-weight: 500;'>বিভিন্ন জেলায় রিয়েল-টাইম মূল্য বিশ্লেষণ করে <b>সর্বোচ্চ লাভ</b> এর স্থান খুঁজে বের করুন।</h3>", unsafe_allow_html=True)
    st.divider()

    # Select Crop
    all_crops = sorted(price_df['Crop_Name'].unique())
    # Create Bengali display names
    all_crops_display = {crop: translate_bn(crop, crop_translation) for crop in all_crops}
    target_crop_bn = st.selectbox("🔍 আপনি কোন ফসল বিক্রি করতে চান?", 
                                   options=list(all_crops_display.values()),
                                   format_func=lambda x: x)
    # Get English name for data filtering
    target_crop = [k for k, v in all_crops_display.items() if v == target_crop_bn][0]

    # Logic: Get the latest price for this crop from every district
    latest_date_in_db = price_df['Price_Date'].max()
    
    # Filter data (taking a 60-day window to ensure we find recent records)
    recent_data = price_df[
        (price_df['Crop_Name'] == target_crop) & 
        (price_df['Price_Date'] >= latest_date_in_db - datetime.timedelta(days=60))
    ]

    # Get the single latest entry for each district
    latest_prices_by_district = recent_data.sort_values('Price_Date').groupby('District_Name').tail(1)

    if not latest_prices_by_district.empty:
        # Find the max price district
        best_market = latest_prices_by_district.sort_values('Price_Tk_kg', ascending=False).iloc[0]
        max_price = best_market['Price_Tk_kg']
        best_district = best_market['District_Name']

        # Display Recommendation with stunning design
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 2rem; border-radius: 15px; color: white; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.3); margin: 1rem 0;'>
            <h2 style='color: white !important; margin: 0; font-size: 2rem;'>🏆 শীর্ষ সুপারিশ</h2>
            <p style='color: white !important; font-size: 1.3rem; margin-top: 1rem;'>
                আপনার <b>{translate_bn(target_crop, crop_translation)}</b> <b style='font-size: 1.5rem;'>{translate_bn(best_district, district_translation)}</b> এ পাঠান!
            </p>
            <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                <p style='color: white !important; margin: 0; font-size: 0.9rem;'>{translate_bn(best_district, district_translation)} এ সর্বোচ্চ মূল্য</p>
                <p style='color: white !important; margin: 0; font-size: 2.5rem; font-weight: 700;'>৳ {to_bengali_number(f'{max_price:.2f}')} / কেজি</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visualization: Bar Chart Comparison
        st.subheader("জেলাগুলিতে মূল্য তুলনা")
        
        fig_bar = px.bar(
            latest_prices_by_district.sort_values('Price_Tk_kg', ascending=True),
            x='Price_Tk_kg',
            y='District_Name',
            orientation='h',
            title=f"{translate_bn(target_crop, crop_translation)} এর বর্তমান বাজার মূল্য",
            labels={'Price_Tk_kg': 'মূল্য (টাকা/কেজি)', 'District_Name': 'জেলা'},
            color='Price_Tk_kg',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.warning("এই ফসলের জন্য সাম্প্রতিক বাজার তথ্য পাওয়া যায়নি।")

# -----------------------------------------------------------------------------
# 6. MODULE 3: SOIL & CROP ADVISOR
# -----------------------------------------------------------------------------
elif menu == "🌱 মাটি ও ফসল পরামর্শদাতা":
    st.title("🌱 বুদ্ধিমান ফসল পরামর্শদাতা")
    st.markdown("<h3 style='color: #1a1a1a; font-weight: 500;'>মাটির স্বাস্থ্য এবং ঐতিহাসিক উৎপাদন তথ্যের উপর ভিত্তি করে বৈজ্ঞানিক সুপারিশ।</h3>", unsafe_allow_html=True)
    st.divider()

    # Input: Select District
    soil_districts = sorted(soil_df['District_Name'].unique())
    soil_district_display = {dist: translate_bn(dist, district_translation) for dist in soil_districts}
    target_district_bn = st.selectbox("📍 আপনার খামারের অবস্থান নির্বাচন করুন", 
                                       options=list(soil_district_display.values()),
                                       format_func=lambda x: x)
    target_district = [k for k, v in soil_district_display.items() if v == target_district_bn][0]

    # --- Soil Analysis ---
    # Get soil data for the selected district (taking the first record found)
    soil_record = soil_df[soil_df['District_Name'] == target_district].iloc[0]

    st.subheader(f"🧪 মাটির স্বাস্থ্য রিপোর্ট: {translate_bn(target_district, district_translation)}")
    
    # Display Soil Metrics in Columns with enhanced styling
    st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌍 মাটির ধরন", translate_bn(soil_record['Soil_Type'], soil_translation))
    c2.metric("⚗️ পিএইচ মাত্রা", to_bengali_number(f"{soil_record['pH_Level']:.2f}"))
    c3.metric("🧬 নাইট্রোজেন (N)", f"{to_bengali_number(f'{soil_record['Nitrogen_Content_kg_ha']:.1f}')} কেজি/হেক্টর")
    c4.metric("🌿 জৈব পদার্থ", f"{to_bengali_number(f'{soil_record['Organic_Matter_Percent']:.1f}')}%")

    # --- Crop Recommendation ---
    st.subheader("🌾 উচ্চ ফলনের জন্য সুপারিশকৃত ফসল")
    st.markdown(f"<p style='color: #1a1a1a; font-size: 1.1rem;'><b>{translate_bn(target_district, district_translation)}</b> এর মাটি বিশ্লেষণ এবং ঐতিহাসিক উৎপাদন তথ্যের উপর ভিত্তি করে নিম্নলিখিত ফসলগুলি সুপারিশ করা হয়:</p>", unsafe_allow_html=True)

    # Logic: Find crops with highest yield (Quintals/Hectare) in this district
    district_prod = prod_df[prod_df['District_Name'] == target_district]
    
    # Group by crop and get average yield (in case of multiple seasons)
    top_crops = district_prod.groupby('Crop_Name')['Yield_Quintals_per_Ha'].mean().sort_values(ascending=False).head(5)

    # Display Top 5 Crops with beautiful cards
    for idx, (crop, yield_val) in enumerate(top_crops.items(), 1):
        crop_bn = translate_bn(crop, crop_translation)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.2rem; border-radius: 10px; color: white; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.8rem 0;'>
            <h3 style='color: white !important; margin: 0; display: flex; align-items: center;'>
                <span style='background: rgba(255,255,255,0.3); padding: 0.3rem 0.8rem; 
                             border-radius: 50%; margin-right: 1rem; font-size: 1.2rem;'>{to_bengali_number(idx)}</span>
                ✅ {crop_bn}
            </h3>
            <p style='color: white !important; font-size: 1rem; margin: 0.5rem 0 0 3rem;'>
                গড় ফলন: <b style='font-size: 1.2rem;'>{to_bengali_number(f'{yield_val:.1f}')}</b> কুইন্টাল/হেক্টর
            </p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; padding: 2rem;'>"
    "<h4 style='color: #1a1a1a; margin: 0;'>Built for <b style='color: #667eea;'>Million Minds for Bangladesh AI Build-a-thon</b></h4>"
    "<p style='color: #2c3e50; margin-top: 0.5rem;'>Powered by Python & Streamlit | 🌾 Empowering Farmers with AI</p>"
    "</div>",
    unsafe_allow_html=True
)