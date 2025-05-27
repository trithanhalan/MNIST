#!/bin/bash

# Create Python 3.9+ virtual environment in ./venv
echo "Creating Python 3.9+ virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Print instructions
echo ""
echo "Setup complete!"
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo "To run the EDA notebook:"
echo "  jupyter notebook notebooks/EDA.ipynb"
echo "To run the Modeling notebook:"
echo "  jupyter notebook notebooks/Modeling.ipynb"
echo "To run the Streamlit app:"
echo "  streamlit run streamlit_app/streamlit_app.py" 