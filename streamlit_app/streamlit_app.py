"""
streamlit_app.py
----------------
Streamlit web app for MNIST Handwritten Digit Prediction.
Features EDA, model training, and digit prediction from user input.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from streamlit_drawable_canvas import st_canvas
import os
from PIL import Image

# --- Utility Functions ---
@st.cache_resource
def load_data():
    """Load MNIST data and return train/test splits."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

@st.cache_resource
def load_trained_model():
    """Load the trained Keras model from disk."""
    model_path = os.path.join('models', 'mnist_cnn.h5')
    return load_model(model_path)

def build_model():
    """Build a new CNN model (same as in Modeling notebook)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
    model = Sequential([
        Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """Train the model and return the history object."""
    from tensorflow.keras.callbacks import Callback
    class StreamlitCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            st.session_state['train_logs'].append(logs)
            st.session_state['current_epoch'] = epoch + 1
    if 'train_logs' not in st.session_state:
        st.session_state['train_logs'] = []
    if 'current_epoch' not in st.session_state:
        st.session_state['current_epoch'] = 0
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[StreamlitCallback()]
    )
    return history

def predict_digit(model, img):
    """Predict digit and probabilities for a single 28x28 image."""
    img = img.astype('float32') / 255.0
    if img.shape != (28,28):
        img = np.resize(img, (28,28))
    img = img[np.newaxis, ..., np.newaxis]
    preds = model.predict(img)
    top3 = preds[0].argsort()[-3:][::-1]
    return [(i, preds[0][i]) for i in top3]

# --- Sidebar ---
st.sidebar.title('MNIST Digit Predictor')
epochs = st.sidebar.slider('Epochs', 1, 20, 10)
batch_size = st.sidebar.selectbox('Batch Size', [32, 64, 128], index=1)

# --- Main Tabs ---
tabs = st.tabs(["EDA", "Train", "Predict"])

# --- EDA Tab ---
with tabs[0]:
    st.header('Exploratory Data Analysis')
    (X_train, y_train), (X_test, y_test) = load_data()
    st.write(f"Train samples: {X_train.shape[0]}")
    st.write(f"Test samples: {X_test.shape[0]}")
    # Show random digits
    idxs = np.random.choice(len(X_train), 20, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(10,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[idxs[i]], cmap='gray')
        ax.set_title(f"{y_train[idxs[i]]}")
        ax.axis('off')
    st.pyplot(fig)
    # Class distribution
    fig2, ax2 = plt.subplots()
    sns.countplot(x=y_train, ax=ax2)
    ax2.set_title('Digit Class Distribution (Train)')
    st.pyplot(fig2)

# --- Train Tab ---
with tabs[1]:
    st.header('Train Model')
    st.write('Train a CNN on MNIST. Model will be saved to models/mnist_cnn.h5.')
    if st.button('Train Model'):
        X_train_, y_train_ = X_train.astype('float32')/255.0, to_categorical(y_train, 10)
        X_test_, y_test_ = X_test.astype('float32')/255.0, to_categorical(y_test, 10)
        X_train_ = X_train_[..., np.newaxis]
        X_test_ = X_test_[..., np.newaxis]
        model = build_model()
        st.session_state['train_logs'] = []
        st.session_state['current_epoch'] = 0
        history = train_model(model, X_train_, y_train_, X_test_, y_test_, epochs, batch_size)
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save('models/mnist_cnn.h5')
        st.success('Training complete! Model saved.')
    # Live plot
    if 'train_logs' in st.session_state and st.session_state['train_logs']:
        logs = st.session_state['train_logs']
        loss = [l['loss'] for l in logs]
        val_loss = [l['val_loss'] for l in logs]
        acc = [l['accuracy'] for l in logs]
        val_acc = [l['val_accuracy'] for l in logs]
        fig, ax = plt.subplots(1,2,figsize=(12,4))
        ax[0].plot(loss, label='Train Loss')
        ax[0].plot(val_loss, label='Val Loss')
        ax[0].legend()
        ax[0].set_title('Loss')
        ax[1].plot(acc, label='Train Acc')
        ax[1].plot(val_acc, label='Val Acc')
        ax[1].legend()
        ax[1].set_title('Accuracy')
        st.pyplot(fig)

# --- Predict Tab ---
with tabs[2]:
    st.header('Predict a Digit')
    st.write('Draw a digit (28x28) or upload an image. The model will predict the top-3 digits.')
    model = None
    try:
        model = load_trained_model()
    except Exception as e:
        st.warning('Trained model not found. Please train and save the model first.')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Draw a digit')
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        img = None
        if canvas_result.image_data is not None:
            img = canvas_result.image_data[:, :, 0]  # Take R channel
            img = np.array(img)
            img = np.clip(img, 0, 255)
            img = 255 - img  # Invert: white digit on black
            img = img / 255.0
            img = (img * 255).astype(np.uint8)
            img = np.array(Image.fromarray(img).resize((28,28)))
    with col2:
        st.subheader('Or upload a 28x28 PNG')
        uploaded = st.file_uploader('Upload digit image', type=['png'])
        if uploaded is not None:
            img = Image.open(uploaded).convert('L').resize((28,28))
            img = np.array(img)
    if model and img is not None:
        top3 = predict_digit(model, img)
        st.image(img, width=100, caption='Input Digit')
        st.write('### Top-3 Predictions:')
        for i, prob in top3:
            st.write(f"Digit {i}: {prob*100:.2f}%") 