# 🌾 Agri-Smart BD - AI-Powered Farm Intelligence Platform

A comprehensive agricultural intelligence platform for Bangladesh farmers, featuring AI-powered price forecasting, smart market recommendations, and soil-based crop advisory.

## 🚀 Features

### 📊 AI Price Forecasting
- Machine Learning-based price prediction using Random Forest algorithm
- 30-day price forecasts for various crops
- Historical price trend analysis
- Smart selling recommendations

### 💰 Best Market Finder
- Real-time price comparison across districts
- Identifies highest-profit markets
- Interactive visualizations for market analysis

### 🌱 Soil & Crop Advisor
- Soil health analysis (pH, nutrients, organic matter)
- Scientific crop recommendations based on soil type
- Historical yield data for informed decisions

## 🎨 UI Features
- Beautiful gradient design with purple theme
- Responsive layout with professional styling
- Interactive Plotly charts
- Clean and modern interface

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/NawrizTurjo/Agri-Price-Pred-millionX.git
cd Agri-Price-Pred-millionX
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Agri-Price-Pred-millionX/
├── app.py                          # Main Streamlit application
├── convert.py                      # Data conversion script
├── requirements.txt                # Python dependencies
├── bd_crop_price_data.csv         # Bangladesh crop price data
├── bd_crop_production_data.csv    # Bangladesh production data
├── bd_soil_analysis_data.csv      # Bangladesh soil data
├── bd_water_usage_data.csv        # Bangladesh water usage data
└── README.md                      # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.13+**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (Random Forest)
- **Plotly** - Interactive visualizations

## 👥 Team

**Million Minds** - AI Build-a-thon 2025

## 📄 License

This project was built for the Million Minds for Bangladesh AI Build-a-thon 2025.

## 🙏 Acknowledgments

- Data converted from Indian agricultural datasets to Bangladesh context
- Built with ❤️ for Bangladesh farmers
