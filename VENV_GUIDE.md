# Virtual Environment Configuration Guide

## 🐍 Virtual Environment Setup

This project uses a Python virtual environment (`.venv`) to ensure consistent dependencies and avoid conflicts.

### ✅ Current Setup Status
- ✅ Virtual environment created at `.venv/`
- ✅ All required packages installed
- ✅ VS Code configured to use `.venv`
- ✅ Run script updated to use `.venv`
- ✅ Streamlit app tested and working

### 🚀 Quick Start Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the Streamlit app
streamlit run app.py

# Or use the provided script
./run.sh
```

### 📦 Package Management

```bash
# Install new packages
source .venv/bin/activate
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

### 🔧 Development Commands

```bash
# All commands should be run with virtual environment activated
source .venv/bin/activate

# Test model loading
python -c "from utils.recommendation_modeling import load_trained_models; print('Models loaded:', len(load_trained_models()[0]))"

# Run specific components
python -c "from utils.company_selection import load_company_data_for_picker; print('Data loaded successfully')"

# Check package versions
python -c "import streamlit, pandas, sklearn; print('All core packages available')"
```

### 🎯 VS Code Integration

The `.vscode/settings.json` file is configured to:
- Use `.venv/bin/python` as the default interpreter
- Automatically activate the virtual environment in terminals
- Set up proper linting and formatting

### 📝 Important Notes

1. **Always use `.venv`**: All Python commands should be run within the virtual environment
2. **Terminal Setup**: New terminals in VS Code will automatically activate `.venv`
3. **Package Installation**: Only install packages within the virtual environment
4. **Streamlit**: Run `streamlit run app.py` from within `.venv`

### 🐛 Troubleshooting

If you encounter issues:

```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Test the setup
python -c "import streamlit; print('Setup successful')"
```

### 📊 Verified Components

- ✅ Streamlit app runs on http://localhost:8501
- ✅ Model loading works (5/7 models functional)
- ✅ About page renders correctly with HTML/CSS
- ✅ Company data loading and analysis features
- ✅ Interactive dashboards and visualizations
- ✅ All utils functions work properly
