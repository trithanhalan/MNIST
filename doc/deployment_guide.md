# MNIST Handwritten Digit Predictor - Deployment Guide

This document provides a step-by-step guide for deploying the MNIST Handwritten Digit Predictor application, primarily focusing on deployment using Streamlit Cloud.

## Phase 1: Local Setup and Model Training

This phase prepares your project locally and generates the necessary model file (`models/mnist_cnn.h5`).

1.  **Clone the Project Repository (If applicable):**
    *   If your project is already in a Git repository, clone it:
        ```bash
        git clone <your_repo_url>
        ```
    *   If you just generated the files locally, you'll need to initialize a Git repository later (Step 6).
    *   Navigate into the project directory. Based on your previous request, this might be:
        ```bash
        cd ~/Desktop/MNIST_Digit_Predictor
        ```

2.  **Run the Setup Script:**
    *   This script creates a Python virtual environment and installs all dependencies listed in `requirements.txt`.
    *   Execute the script from the project root:
        ```bash
        bash setup.sh
        ```
    *   Follow the on-screen instructions. It will print commands to activate the virtual environment and run the notebooks/app.

3.  **Activate the Virtual Environment:**
    *   Activate the environment created by `setup.sh`:
        ```bash
        source venv/bin/activate
        ```
    *   You should see `(venv)` at the beginning of your terminal prompt, indicating the environment is active.

4.  **Run the EDA Notebook (Optional but Recommended):**
    *   To explore the data, start Jupyter Notebook:
        ```bash
        jupyter notebook notebooks/EDA.ipynb
        ```
    *   This will open the notebook in your web browser. Run through the cells to see the dataset statistics, sample images, and class distribution.

5.  **Run the Modeling Notebook (Crucial for Model File):**
    *   Open the modeling notebook:
        ```bash
        jupyter notebook notebooks/Modeling.ipynb
        ```
    *   **Important:** Run through *all* cells in this notebook. This will:
        *   Load and preprocess the data.
        *   Define the CNN model.
        *   Train the model using the specified `epochs` and `batch_size`.
        *   **Save the trained model to `models/mnist_cnn.h5`**. This file is essential for the Streamlit app to run without retraining on deployment.
        *   Evaluate the model and display results.
    *   Make sure the `models/mnist_cnn.h5` file is successfully created in your project's `models/` directory.

6.  **Verify Local Streamlit App (Optional but Recommended):**
    *   While the virtual environment is active, run the Streamlit app locally to ensure everything works before deploying:
        ```bash
        streamlit run streamlit_app/streamlit_app.py
        ```
    *   This will open the app in your web browser. Test the EDA tab, check if the "Train Model" button works (it will retrain with sidebar settings), and importantly, verify that the Predict tab loads the saved model (`models/mnist_cnn.h5`) and allows you to draw or upload a digit for prediction.

## Phase 2: Preparing for Deployment (Using Git and GitHub)

Streamlit Cloud deploys directly from a Git repository (like GitHub, GitLab, or Bitbucket).

7.  **Initialize a Git Repository (If not already in one):**
    *   If your project was not cloned from a repo, initialize one in your project root:
        ```bash
        git init
        ```
    *   Add all project files to the staging area:
        ```bash
        git add .
        ```
    *   Commit the changes:
        ```bash
        git commit -m "Initial commit: Add project files, notebooks, and trained model"
        ```
    *   **Important:** Make sure the `models/mnist_cnn.h5` file is included in your commit. Since it's a saved model file, it might be large, but for typical MNIST CNNs, it's usually manageable for Git and crucial for Streamlit Cloud deployment. Ensure you do *not* have `models/` in your `.gitignore` file if one exists.

8.  **Create a Remote Repository (e.g., on GitHub):**
    *   Go to your preferred Git hosting service (GitHub.com, GitLab.com, etc.).
    *   Create a new empty repository. **Do NOT initialize it with a README or license file** as you already have files locally.
    *   Follow the instructions provided by the hosting service to link your local repository to the new remote one and push your code. It will look something like this (replace with your specific repo details):
        ```bash
        git remote add origin <remote_repository_url>
        git branch -M main # Or 'master' depending on your preference/service default
        git push -u origin main
        ```
    *   Verify that all your project files, including the `models/mnist_cnn.h5` file, `requirements.txt`, `streamlit_app/streamlit_app.py`, and `.streamlit/config.toml` are present in your online repository.

## Phase 3: Deployment to Streamlit Cloud

Now you will use Streamlit Cloud to deploy the app from your GitHub repository.

9.  **Sign up/Log in to Streamlit Cloud:**
    *   Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
    *   Sign up or log in, typically using your GitHub account.

10. **Deploy a New App:**
    *   Once logged in, click the "New app" button or "Deploy an app".
    *   Select "From existing repo".

11. **Connect to Your Repository:**
    *   Choose your repository from the list. If you logged in with GitHub, it should show your GitHub repositories. You might need to authorize Streamlit to access your repositories.
    *   Select the branch you want to deploy from (usually `main` or `master`).

12. **Configure Deployment Settings:**
    *   **Repository:** Your chosen GitHub repository.
    *   **Branch:** The branch (e.g., `main`).
    *   **Main file path:** Enter the path to your Streamlit app script relative to the repository root. This should be:
        ```
        streamlit_app/streamlit_app.py
        ```
    *   **Python version:** Select a Python version compatible with your `requirements.txt`. Python 3.9 is a safe choice as specified, but you might also select 3.10 or 3.11 depending on availability and dependency compatibility (TensorFlow 2.12 works with 3.9-3.11).
    *   **Advanced settings (Optional but check):**
        *   **Secrets:** You don't need any secrets for this project.
        *   **Package management:** Ensure this is set to automatically use `requirements.txt`. Streamlit Cloud automatically detects and uses this file.
        *   **Working directory:** Ensure this is set to the root of your repository (`.`) so the app can find the `models/` directory correctly. This is usually the default.

13. **Deploy the App:**
    *   Click the "Deploy!" button.
    *   Streamlit Cloud will now:
        *   Clone your repository.
        *   Set up the specified Python environment.
        *   Install dependencies from `requirements.txt`.
        *   Run your `streamlit_app/streamlit_app.py` script.

14. **Monitor Deployment and Access App:**
    *   You will see the deployment logs in the Streamlit Cloud interface. Monitor these logs for any errors during setup or execution. Common issues include incorrect file paths, missing dependencies (if not in `requirements.txt`), or problems loading the model file if it wasn't committed correctly.
    *   Once deployment is successful, you will be provided with a public URL for your application.

15. **Update README.md:**
    *   Go back to your local project directory.
    *   Update the `README.md` file to replace the placeholder `[Add your deployed app link here]` with the actual URL of your deployed Streamlit app.
    *   Commit and push this change to your GitHub repository:
        ```bash
        git add README.md
        git commit -m "Update README with Streamlit Cloud link"
        git push origin main # Or your branch name
        ```
    *   Streamlit Cloud will automatically detect this commit and redeploy the app (this usually takes a few minutes).

## Phase 4: Verification

16. **Test the Deployed App:**
    *   Open the public URL of your Streamlit app in a web browser.
    *   Navigate through the tabs.
    *   Verify the EDA tab displays correctly.
    *   Verify the Predict tab loads the model and allows you to draw/upload digits and see predictions.
    *   (Optional) You can try the "Train Model" button on the deployed app, but be aware that any model trained and saved *on Streamlit Cloud* will be ephemeral and will not persist between deployments or app restarts unless you set up persistent storage (which is not covered in this basic guide). The primary model used for prediction should be the one you trained locally and committed to Git. 