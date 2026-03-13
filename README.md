# Brain Tumor MRI Classifier

This repository contains a brain tumor MRI classification model and a modern FastAPI web application with a glassmorphism HTML/CSS/JS frontend to test the model on new MRI images.

## Dataset
The model was trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

## Setup Instructions

### 1. Create an Anaconda Environment

It is recommended to use an Anaconda environment to manage dependencies:

```bash
conda create -n brain-tumor-env python=3.10 -y
conda activate brain-tumor-env
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Application

Ensure `Pretrained_model.keras` is located in the root project directory alongside `main.py`. Then, start the Uvicorn development server:

```bash
python -m uvicorn main:app --reload
```

This will launch the backend and serve the application UI. Open your web browser and navigate to:
**http://127.0.0.1:8000**

You can then upload MRI images directly through the drag-and-drop web interface to automatically classify them.

## Note on Pre-trained Model
The pre-trained model `Pretrained_model.keras` is ignored in Git due to its large size (exceeding GitHub's 100MB limit). You will need to keep this model locally in the project directory when running the app.
