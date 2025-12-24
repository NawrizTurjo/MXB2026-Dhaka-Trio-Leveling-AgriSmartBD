import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
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
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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
        border: 2px solid #11998e !important;
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
        background-color: #11998e !important;
        color: #ffffff !important;
    }
    
    /* Selected option in dropdown */
    [aria-selected="true"] {
        background-color: #38ef7d !important;
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f5132 0%, #198754 100%);
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
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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
with st.sidebar.expander("ℹ️ তথ্য এবং ডেটা উৎস"):
    st.write("**ডেটা সোর্স:** এই প্রোটোটাইপটি ঐতিহাসিক কৃষি ডেটা এবং আবহাওয়ার প্যাটার্ন ব্যবহার করে তৈরি করা হয়েছে।")
    st.write("**গোপনীয়তা:** এটি শুধুমাত্র একটি ডেমো। কোনো ব্যক্তিগত তথ্য সংরক্ষণ করা হয় না।")
    st.write("**Team:** Trio Leveling | Build-a-thon 2025")

# -----------------------------------------------------------------------------
# 4. MODULE 1: AI PRICE FORECASTING
# -----------------------------------------------------------------------------
if menu == "📊 মূল্য পূর্বাভাস (এআই)":
    st.title("📊 এআই চালিত মূল্য পূর্বাভাস")
    st.markdown("### মেশিন লার্নিং ব্যবহার করে ৩০ দিনের আগাম মূল্যের পূর্বাভাস এবং অনিশ্চয়তা বিশ্লেষণ।")
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
        # --- MACHINE LEARNING SECTION (SMARTER VERSION) ---
        
        # 1. Feature Engineering: Add Seasonality (ঋতুভিত্তিক বৈচিত্র্য যোগ করা)
        filtered_df['Month'] = filtered_df['Price_Date'].dt.month
        filtered_df['Week'] = filtered_df['Price_Date'].dt.isocalendar().week
        filtered_df['Year'] = filtered_df['Price_Date'].dt.year
        filtered_df['Date_Ordinal'] = filtered_df['Price_Date'].map(datetime.datetime.toordinal)
        
        # 2. Define Features (X) and Target (y)
        # We use Month and Week to capture seasonal trends
        X = filtered_df[['Date_Ordinal', 'Month', 'Week', 'Year']]
        y = filtered_df['Price_Tk_kg']
        
        # 3. Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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
        
        # 5. Predict with Confidence Intervals
        X_future = future_data[['Date_Ordinal', 'Month', 'Week', 'Year']]
        
        # Get predictions from all individual trees to calculate variance
        # Convert to numpy array to avoid feature name warnings
        predictions = [tree.predict(X_future.values) for tree in model.estimators_]
        predictions = np.array(predictions)
        
        # Mean prediction
        future_prices = np.mean(predictions, axis=0)
        # Standard Deviation (Volatility)
        std_dev = np.std(predictions, axis=0)
        
        future_data['Predicted_Price'] = future_prices
        future_data['Upper_Bound'] = future_prices + (std_dev * 1.5)  # 1.5 sigma
        future_data['Lower_Bound'] = future_prices - (std_dev * 1.5)

        # --- VISUALIZATION (Plotly Graph Objects for Confidence Bands) ---
        st.subheader(f"মূল্য প্রবণতা ও ঝুঁকি বিশ্লেষণ: {translate_bn(selected_crop, crop_translation)}")
        
        fig = go.Figure()
        
        # Historical Line
        fig.add_trace(go.Scatter(
            x=filtered_df['Price_Date'], y=filtered_df['Price_Tk_kg'],
            mode='lines', name='ঐতিহাসিক তথ্য', line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast Line
        fig.add_trace(go.Scatter(
            x=future_data['Price_Date'], y=future_data['Predicted_Price'],
            mode='lines', name='এআই পূর্বাভাস', line=dict(color='#00cc96', width=2)
        ))
        
        # Confidence Interval (Upper + Lower)
        fig.add_trace(go.Scatter(
            x=pd.concat([future_data['Price_Date'], future_data['Price_Date'][::-1]]),
            y=pd.concat([future_data['Upper_Bound'], future_data['Lower_Bound'][::-1]]),
            fill='toself', fillcolor='rgba(0, 204, 150, 0.2)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False,
            name='সম্ভাব্য পরিসীমা'
        ))
        
        fig.update_layout(
            title=f"আগামী ৩০ দিনের সম্ভাব্য মূল্য পরিসীমা (Confidence Interval)",
            xaxis_title="তারিখ", yaxis_title="মূল্য (টাকা/কেজি)",
            hovermode="x unified", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- INSIGHTS & METRICS ---
        current_price = filtered_df.iloc[-1]['Price_Tk_kg']
        avg_price = future_prices.mean()
        trend = "উর্ধ্বমুখী 📈" if avg_price > current_price else "নিম্নমুখী 📉"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("বর্তমান গড় মূল্য", f"৳ {to_bengali_number(f'{current_price:.2f}')}")
        m2.metric("পূর্বাভাস (গড়)", f"৳ {to_bengali_number(f'{avg_price:.2f}')}")
        m3.metric("প্রবণতা", trend)
        
        st.info(f"💡 **বিশ্লেষণ:** এআই মডেলটি {to_bengali_number(100)} টি ডিসিশন ট্রি ব্যবহার করে এই পূর্বাভাস দিয়েছে। হালকা সবুজ অংশটি সম্ভাব্য মূল্যের ওঠানামা নির্দেশ করে।")

        st.markdown("---")
        st.subheader("📲 এসএমএস অ্যালার্ট (Farmer Connect)")
        st.markdown("<small>প্রোটোটাইপ ভার্সন - SMS গেটওয়ে ইন্টিগ্রেশন আসছে শীঘ্রই</small>", unsafe_allow_html=True)
        phone = st.text_input("মোবাইল নম্বর", placeholder="017XXXXXXXX", key="sms_phone")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            send_button = st.button("📩 মূল্য সতর্কতা পাঠান", use_container_width=True)
        if send_button and phone:
            st.success(f"✅ {phone}-এ অ্যালার্ট পাঠানো হয়েছে!")
        elif send_button and not phone:
            st.warning("⚠️ অনুগ্রহ করে একটি মোবাইল নম্বর প্রদান করুন।")

# -----------------------------------------------------------------------------
# 5. MODULE 2: BEST MARKET FINDER
# -----------------------------------------------------------------------------
elif menu == "💰 সেরা বাজার খুঁজুন":
    st.title("💰 স্মার্ট বাজার ও লাভ ক্যালকুলেটর")
    st.markdown("### পরিবহন খরচ বাদ দিয়ে **প্রকৃত লাভ (Net Profit)** কোথায় বেশি তা জানুন।")
    st.divider()

    all_crops = sorted(price_df['Crop_Name'].unique())
    all_crops_display = {crop: translate_bn(crop, crop_translation) for crop in all_crops}
    target_crop_bn = st.selectbox("🔍 ফসল নির্বাচন করুন", options=list(all_crops_display.values()))
    target_crop = [k for k, v in all_crops_display.items() if v == target_crop_bn][0]

    # New Input: Transport Cost
    st.markdown("#### 🚚 পরিবহন খরচ হিসাব করুন")
    transport_cost = st.number_input("প্রতি কেজিতে পরিবহন খরচ কত? (টাকা)", min_value=0.0, value=2.0, step=0.5)

    # Get Data
    latest_date = price_df['Price_Date'].max()
    recent_data = price_df[(price_df['Crop_Name'] == target_crop) & (price_df['Price_Date'] >= latest_date - datetime.timedelta(days=60))]
    market_data = recent_data.sort_values('Price_Date').groupby('District_Name').tail(1).copy()

    if not market_data.empty:
        # Calculate Net Profit
        market_data['Net_Profit'] = market_data['Price_Tk_kg'] - transport_cost
        best_market = market_data.sort_values('Net_Profit', ascending=False).iloc[0]
        
        # Recommendation Card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;'>
            <h2 style='color: white !important; margin: 0;'>🏆 সেরা বাজার: {translate_bn(best_market['District_Name'], district_translation)}</h2>
            <p style='color: white !important; font-size: 1.2rem; margin-top: 10px;'>
                বিক্রয় মূল্য: ৳ {to_bengali_number(f"{best_market['Price_Tk_kg']:.2f}")} | 
                খরচ: ৳ {to_bengali_number(f"{transport_cost:.2f}")}
            </p>
            <h1 style='color: white !important; margin: 0; font-size: 3rem;'>
                নিট লাভ: ৳ {to_bengali_number(f"{best_market['Net_Profit']:.2f}")} <span style='font-size:1rem'>/কেজি</span>
            </h1>
        </div>
        """, unsafe_allow_html=True)

        # Bar Chart showing Profit
        fig = px.bar(
            market_data.sort_values('Net_Profit', ascending=True),
            x='Net_Profit', y='District_Name', orientation='h',
            title=f"বিভিন্ন জেলায় সম্ভাব্য নিট লাভ (পরিবহন খরচ বাদে)",
            labels={'Net_Profit': 'নিট লাভ (টাকা/কেজি)', 'District_Name': 'জেলা'},
            color='Net_Profit', color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("তথ্য পাওয়া যায়নি।")

# -----------------------------------------------------------------------------
# 6. MODULE 3: SOIL & CROP ADVISOR
# -----------------------------------------------------------------------------
elif menu == "🌱 মাটি ও ফসল পরামর্শদাতা":
    st.title("🌱 বুদ্ধিমান ফসল পরামর্শদাতা")
    st.markdown("### মাটির গুণাগুণ বিশ্লেষণ করে বৈজ্ঞানিক চাষাবাদ পরামর্শ।")
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

    # Soil Dashboard
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌍 মাটির ধরন", translate_bn(soil_record['Soil_Type'], soil_translation))
    c2.metric("⚗️ পিএইচ মাত্রা", to_bengali_number(f"{soil_record['pH_Level']:.2f}"))
    c3.metric("🧬 নাইট্রোজেন (N)", f"{to_bengali_number(f'{soil_record['Nitrogen_Content_kg_ha']:.1f}')} কেজি/হেক্টর")
    c4.metric("🌿 জৈব পদার্থ", f"{to_bengali_number(f'{soil_record['Organic_Matter_Percent']:.1f}')}%")

    st.subheader("🌾 সুপারিশকৃত ফসল ও কারণ")
    
    dist_prod = prod_df[prod_df['District_Name'] == target_district]
    top_crops = dist_prod.groupby('Crop_Name')['Yield_Quintals_per_Ha'].mean().sort_values(ascending=False).head(5)

    for idx, (crop, yield_val) in enumerate(top_crops.items(), 1):
        soil_type_bn = translate_bn(soil_record['Soil_Type'], soil_translation)
        
        # Simple Logic for "Reasoning" Text
        reason = f"এই অঞ্চলের <b>{soil_type_bn}</b> এবং আবহাওয়া <b>{translate_bn(crop, crop_translation)}</b> চাষের জন্য অত্যন্ত উপযোগী।"
        
        st.markdown(f"""
        <div style='background: white; border-left: 5px solid #11998e; padding: 1.2rem; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
            <h3 style='margin:0; color: #11998e !important;'>#{idx} {translate_bn(crop, crop_translation)}</h3>
            <p style='margin: 5px 0 0 0; font-size: 0.95rem;'><b>ঐতিহাসিক ফলন:</b> {to_bengali_number(f'{yield_val:.1f}')} কুইন্টাল/হেক্টর</p>
            <p style='margin: 5px 0 0 0; font-size: 0.9rem; color: #555;'>✅ <b>কারণ:</b> {reason}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><hr><div style='text-align: center; color: #555;'>Agri-Smart BD | Built for AI Build-a-thon 2025</div>", unsafe_allow_html=True)