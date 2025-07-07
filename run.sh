#!/bin/bash

# Company Recommendation System - Run Script
echo "🏢 Starting Company Recommendation System..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if data file exists
if [ ! -f "../Du lieu cung cap/Overview_Companies.xlsx" ]; then
    echo "⚠️  Warning: Data file not found at '../Du lieu cung cap/Overview_Companies.xlsx'"
    echo "Please ensure the data file is in the correct location."
fi

# Run the Streamlit app
echo "🚀 Starting Streamlit application..."
streamlit run app.py

echo "✅ Application started! Open your browser to view the app."
