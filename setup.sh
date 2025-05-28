#!/bin/bash

# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models notebooks streamlit_app reports

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then you can:"
echo "1. Run EDA notebook: jupyter notebook notebooks/EDA.ipynb"
echo "2. Run Modeling notebook: jupyter notebook notebooks/Modeling.ipynb"
echo "3. Launch Streamlit app: streamlit run streamlit_app/streamlit_app.py" 