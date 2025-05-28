"""
streamlit_app.py
----------------
Streamlit web app for MNIST Handwritten Digit Prediction.
Features EDA, model training, and digit prediction from user input.
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import time
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢",
    layout="wide"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess MNIST data"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test), (x_train_cnn, x_test_cnn), (y_train_cat, y_test_cat)

# Cache model loading
@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        mlp_model = load_model('models/mlp_baseline.h5')
        cnn_model = load_model('models/mnist_cnn.h5')
        return mlp_model, cnn_model
    except:
        return None, None

def build_mlp():
    """Build MLP model"""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def build_cnn():
    """Build CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs, batch_size):
    """Train model and return history"""
    history = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2,
                       verbose=0)
    return history

def compare_models(mlp_model, cnn_model, x_test, x_test_cnn, y_test):
    """Compare model performance"""
    # Evaluate models
    mlp_test_loss, mlp_test_acc = mlp_model.evaluate(x_test, y_test, verbose=0)
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
    
    # Get predictions
    mlp_pred = np.argmax(mlp_model.predict(x_test), axis=1)
    cnn_pred = np.argmax(cnn_model.predict(x_test_cnn), axis=1)
    
    # Calculate metrics
    mlp_report = classification_report(y_test, mlp_pred, output_dict=True)
    cnn_report = classification_report(y_test, cnn_pred, output_dict=True)
    
    # Create summary
    summary = pd.DataFrame({
        'Model': ['MLP', 'CNN'],
        'Test Accuracy': [mlp_test_acc, cnn_test_acc],
        'Test Loss': [mlp_test_loss, cnn_test_loss],
        'F1 Score (weighted avg)': [mlp_report['weighted avg']['f1-score'],
                                  cnn_report['weighted avg']['f1-score']]
    })
    
    return summary, mlp_pred, cnn_pred

def predict_digit(model, image):
    """Predict digit from image and return prediction and latency"""
    # Preprocess image
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    
    if len(img.shape) == 3:
        img = img[:, :, 0]  # Take first channel if RGB
    
    img = img.astype('float32') / 255.0
    
    # Reshape for model
    if isinstance(model, Sequential) and model.layers[0].input_shape[-1] == 1:
        img = img.reshape(1, 28, 28, 1)
    else:
        img = img.reshape(1, 28, 28)
    
    # Measure prediction latency
    start_time = time.time()
    pred = model.predict(img, verbose=0)
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return pred[0], latency

def measure_batch_latency(model, x_test, batch_size=100):
    """Measure average prediction latency on a batch of test data"""
    start_time = time.time()
    model.predict(x_test[:batch_size], verbose=0)
    latency = (time.time() - start_time) * 1000 / batch_size  # Average ms per prediction
    return latency

# Main app
def main():
    st.title("MNIST Handwritten Digit Predictor")
    
    # Load data and models
    (x_train, y_train), (x_test, y_test), (x_train_cnn, x_test_cnn), (y_train_cat, y_test_cat) = load_data()
    mlp_model, cnn_model = load_models()
    
    # Sidebar controls
    st.sidebar.header("Model Training Controls")
    epochs = st.sidebar.slider("Epochs", 1, 20, 10)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
    
    if st.sidebar.button("Train Models"):
        with st.spinner("Training models..."):
            # Train MLP
            mlp_model = build_mlp()
            mlp_start_time = time.time()
            mlp_history = train_model(mlp_model, x_train, y_train_cat, epochs, batch_size)
            mlp_training_time = time.time() - mlp_start_time
            
            # Train CNN
            cnn_model = build_cnn()
            cnn_start_time = time.time()
            cnn_history = train_model(cnn_model, x_train_cnn, y_train_cat, epochs, batch_size)
            cnn_training_time = time.time() - cnn_start_time
            
            # Save models
            mlp_model.save('models/mlp_baseline.h5')
            cnn_model.save('models/mnist_cnn.h5')
            
            st.success("Models trained and saved successfully!")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["EDA", "Train & Compare", "Predict"])
    
    with tab1:
        st.header("Exploratory Data Analysis")
        
        # Display sample images
        st.subheader("Sample Training Images")
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        for i in range(20):
            row, col = i // 5, i % 5
            axes[row, col].imshow(x_train[i], cmap='gray')
            axes[row, col].set_title(f"Label: {y_train[i]}")
            axes[row, col].axis('off')
        st.pyplot(fig)
        
        # Class distribution
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=y_train, ax=ax)
        ax.set_title("Distribution of Digits in Training Set")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        # Pixel statistics
        st.subheader("Pixel Statistics")
        mean_intensity = np.mean(x_train)
        std_intensity = np.std(x_train)
        st.write(f"Mean pixel intensity: {mean_intensity:.4f}")
        st.write(f"Standard deviation of pixel intensity: {std_intensity:.4f}")
    
    with tab2:
        st.header("Model Training and Comparison")
        
        if mlp_model is None or cnn_model is None:
            st.warning("Please train the models first using the sidebar controls.")
        else:
            # Compare models
            summary, mlp_pred, cnn_pred = compare_models(mlp_model, cnn_model, x_test, x_test_cnn, y_test)
            
            # Measure latencies
            mlp_latency = measure_batch_latency(mlp_model, x_test)
            cnn_latency = measure_batch_latency(cnn_model, x_test_cnn)
            
            # Add latency to summary
            summary['Avg Latency (ms)'] = [mlp_latency, cnn_latency]
            
            # Display summary
            st.subheader("Model Performance Summary")
            st.dataframe(summary.style.format({
                'Test Accuracy': '{:.4f}',
                'Test Loss': '{:.4f}',
                'F1 Score (weighted avg)': '{:.4f}',
                'Avg Latency (ms)': '{:.2f}'
            }))
            
            # Plot confusion matrices
            st.subheader("Confusion Matrices")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            sns.heatmap(confusion_matrix(y_test, mlp_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('MLP Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')
            
            sns.heatmap(confusion_matrix(y_test, cnn_pred), annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('CNN Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
            
            st.pyplot(fig)
            
            # Display latency comparison
            st.subheader("Model Latency Comparison")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=summary, x='Model', y='Avg Latency (ms)', ax=ax)
            ax.set_title('Average Prediction Latency per Image')
            ax.set_ylabel('Latency (milliseconds)')
            st.pyplot(fig)
    
    with tab3:
        st.header("Digit Prediction")
        
        if mlp_model is None or cnn_model is None:
            st.warning("Please train the models first using the sidebar controls.")
        else:
            # Image input
            st.subheader("Input Image")
            uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                # Load and preprocess image
                image = Image.open(uploaded_file).convert('L')
                image = image.resize((28, 28))
                
                # Display image
                st.image(image, caption="Uploaded Image", width=200)
                
                # Get predictions and latencies
                mlp_pred, mlp_latency = predict_digit(mlp_model, image)
                cnn_pred, cnn_latency = predict_digit(cnn_model, image)
                
                # Display predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("MLP Predictions")
                    st.write(f"Prediction Latency: {mlp_latency:.2f} ms")
                    pred_df = pd.DataFrame({
                        'Digit': range(10),
                        'Probability': mlp_pred
                    }).sort_values('Probability', ascending=False)
                    st.dataframe(pred_df)
                    
                    # Plot probabilities
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=pred_df, x='Digit', y='Probability', ax=ax)
                    ax.set_title('MLP Prediction Probabilities')
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("CNN Predictions")
                    st.write(f"Prediction Latency: {cnn_latency:.2f} ms")
                    pred_df = pd.DataFrame({
                        'Digit': range(10),
                        'Probability': cnn_pred
                    }).sort_values('Probability', ascending=False)
                    st.dataframe(pred_df)
                    
                    # Plot probabilities
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=pred_df, x='Digit', y='Probability', ax=ax)
                    ax.set_title('CNN Prediction Probabilities')
                    st.pyplot(fig)

if __name__ == "__main__":
    main() 