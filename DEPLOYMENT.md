# Deployment Guide for Agri-Smart BD

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)

1. **Prerequisites:**
   - GitHub account
   - Push this project to a GitHub repository

2. **Steps:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch (main), and main file path (`app.py`)
   - Click "Deploy"

3. **Configuration:**
   - The app will automatically use `requirements.txt`
   - Python version is specified in `.python-version`
   - Streamlit config is in `.streamlit/config.toml`

### Option 2: Heroku

1. **Install Heroku CLI:**
   ```bash
   brew tap heroku/brew && brew install heroku
   ```

2. **Login to Heroku:**
   ```bash
   heroku login
   ```

3. **Create a new Heroku app:**
   ```bash
   heroku create agri-smart-bd
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

### Option 3: Local Deployment

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

### Option 4: Docker

1. **Build the Docker image:**
   ```bash
   docker build -t agri-smart-bd .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 agri-smart-bd
   ```

## 📝 Important Notes

### Data Files Required:
Ensure these CSV files are in the same directory as `app.py`:
- `bd_crop_price_data.csv`
- `bd_crop_production_data.csv`
- `bd_soil_analysis_data.csv`
- `bd_water_usage_data.csv`

### Voice Recognition:
- Voice features require microphone permissions
- May have limited functionality on some cloud platforms
- For production, consider using dedicated speech API services

### Environment Variables (Optional):
Create a `.streamlit/secrets.toml` for sensitive data:
```toml
# Add any API keys or secrets here
```

## 🔧 Troubleshooting

### Issue: App crashes on startup
- Check that all CSV files are present
- Verify Python version compatibility (3.8-3.11)

### Issue: Voice recognition not working
- Microphone permissions may be blocked
- Consider implementing fallback text input

### Issue: Slow loading
- Consider reducing the size of datasets
- Implement pagination for large data displays
- Use caching more aggressively

## 🌐 Post-Deployment

1. Test all features thoroughly
2. Monitor logs for errors
3. Set up analytics (optional)
4. Configure custom domain (optional)

## 📞 Support

For issues or questions:
- GitHub Issues: [Your Repository URL]
- Team: Trio Leveling
- Build-a-thon 2025
