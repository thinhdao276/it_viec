@echo off
echo 🏢 Starting Company Recommendation System...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if data file exists
if not exist "..\Du lieu cung cap\Overview_Companies.xlsx" (
    echo ⚠️  Warning: Data file not found at '../Du lieu cung cap/Overview_Companies.xlsx'
    echo Please ensure the data file is in the correct location.
)

REM Run the Streamlit app
echo 🚀 Starting Streamlit application...
streamlit run app.py

echo ✅ Application started! Open your browser to view the app.
pause
