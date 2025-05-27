# MNIST Handwritten Digit Predictor

This project provides an end-to-end solution for classifying handwritten digits from the MNIST dataset. It includes exploratory data analysis (EDA), a convolutional neural network (CNN) model, and an interactive Streamlit web app for training and prediction. The project is ready for deployment and easy to use.

## Quick Start
```bash
git clone <repo>
cd project_root
bash setup.sh
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/Modeling.ipynb
streamlit run streamlit_app/streamlit_app.py
```

## Folder Structure
- `data/` — (Optional) Store raw or processed data files.
- `models/` — Saved Keras models (e.g., `mnist_cnn.h5`).
- `notebooks/` — Jupyter notebooks for EDA and modeling.
- `streamlit_app/` — Source code for the Streamlit web app.
- `reports/` — Generated reports, figures, or analysis outputs.
- `requirements.txt` — All Python dependencies with exact versions.
- `README.md` — Project overview and instructions.

## Model, Notebooks, and Reports
- The trained model is saved in `models/mnist_cnn.h5` after running the modeling notebook.
- EDA and modeling steps are documented in `notebooks/EDA.ipynb` and `notebooks/Modeling.ipynb`.
- Any generated reports or figures can be found in `reports/`.

## Deployment
- The Streamlit app is in `streamlit_app/streamlit_app.py` and ready for Streamlit Cloud.
- Model loads directly from `models/mnist_cnn.h5` (no retraining required).
- (Optional) Streamlit Cloud link: [Add your deployed app link here] 