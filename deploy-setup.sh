#!/bin/bash

echo "🚀 Starting Agri-Smart BD Deployment Setup..."

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Agri-Smart BD application"
else
    echo "✅ Git repository already initialized"
fi

# Check if requirements.txt exists
if [ -f requirements.txt ]; then
    echo "✅ requirements.txt found"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Check if all CSV files exist
files=("bd_crop_price_data.csv" "bd_crop_production_data.csv" "bd_soil_analysis_data.csv" "bd_water_usage_data.csv")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found"
    else
        echo "⚠️  Warning: $file not found - app may not work correctly"
    fi
done

echo ""
echo "🎉 Deployment files created successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. For Streamlit Cloud (Recommended):"
echo "   - Create a GitHub repository"
echo "   - Push this code: git remote add origin <your-repo-url>"
echo "   - git push -u origin main"
echo "   - Visit https://share.streamlit.io and deploy"
echo ""
echo "2. For Local Testing:"
echo "   - Run: streamlit run app.py"
echo ""
echo "3. For Heroku:"
echo "   - Run: heroku create agri-smart-bd"
echo "   - Run: git push heroku main"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions"
