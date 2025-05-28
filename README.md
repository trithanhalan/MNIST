# MNIST Handwritten Digit Predictor

A comprehensive machine learning project for handwritten digit recognition using the MNIST dataset. This project includes exploratory data analysis, model comparison between MLP and CNN architectures, and an interactive web interface for predictions.

## Dataset
The project uses the [MNIST Handwritten Digits Dataset](https://www.kaggle.com/c/digit-recognizer) (loaded via Keras), which contains 70,000 grayscale images of handwritten digits (28x28 pixels).

## Features

### 1. Model Comparison
- **MLP (Multi-Layer Perceptron)**: A baseline fully-connected neural network
- **CNN (Convolutional Neural Network)**: A more sophisticated architecture with convolutional layers
- **Performance Metrics**:
  - Test accuracy
  - F1 score
  - Prediction latency (milliseconds)
  - Confusion matrices
  - Training time comparison

### 2. Interactive Web Interface
- **EDA Tab**: Visualize dataset statistics and sample images
- **Train & Compare Tab**: Train both models and compare their performance
- **Predict Tab**: Upload or draw digits for real-time predictions
  - Shows predictions from both models
  - Displays prediction probabilities
  - Reports prediction latency for each model

## Quick Start

1. Clone the repository:
```bash
git clone <your_repo_url>
cd MNIST_Digit_Predictor
```

2. Run the setup script:
```bash
bash setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Run the notebooks or launch the app:
```bash
# For EDA:
jupyter notebook notebooks/EDA.ipynb

# For model training and comparison:
jupyter notebook notebooks/Modeling.ipynb

# For the interactive web app:
streamlit run streamlit_app/streamlit_app.py
```

## Project Structure
```
MNIST_Digit_Predictor/
├── data/               # Data storage
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks
│   ├── EDA.ipynb     # Exploratory data analysis
│   └── Modeling.ipynb # Model training and comparison
├── streamlit_app/     # Web application
│   └── streamlit_app.py
├── reports/          # Generated reports and visualizations
├── requirements.txt  # Project dependencies
└── setup.sh         # Setup script
```

## Model Performance

The project compares two neural network architectures:

1. **MLP (Multi-Layer Perceptron)**
   - Architecture: Flatten → Dense(512) → Dropout → Dense(256) → Dropout → Dense(10)
   - Advantages: Simpler architecture, faster training
   - Use cases: When computational resources are limited

2. **CNN (Convolutional Neural Network)**
   - Architecture: Two Conv2D blocks with BatchNorm and MaxPooling → Dense layers
   - Advantages: Better feature extraction, higher accuracy
   - Use cases: When accuracy is the primary concern

Both models are evaluated on:
- Test accuracy
- F1 score
- Prediction latency
- Confusion matrix analysis

## Deployment
The application is deployed on Streamlit Cloud. You can access it at: [Add your deployed app link here]

## Requirements
- Python 3.9+
- TensorFlow 2.12.0
- Streamlit 1.22.0
- Other dependencies listed in `requirements.txt`

## Contributing
Feel free to submit issues and enhancement requests!