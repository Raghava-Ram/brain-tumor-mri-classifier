# Brain Tumor MRI Classifier

This repository contains a brain tumor MRI classification model and three different web application frontends (FastAPI, Gradio, and Streamlit) to test the model on new MRI images.

## Model Download

The trained EfficientNet model used in this project is hosted on Hugging Face. The application is configured to download this model automatically using the `huggingface_hub` via the `hf_hub_download` function, meaning you do not need to manually download the model file to run the applications.

You can download the model via the browser here:
[https://huggingface.co/Raghava-Ram/brain-tumor-efficientnet](https://huggingface.co/Raghava-Ram/brain-tumor-efficientnet)

**Direct model file:**
[https://huggingface.co/Raghava-Ram/brain-tumor-efficientnet/resolve/main/pretrained_model.keras](https://huggingface.co/Raghava-Ram/brain-tumor-efficientnet/resolve/main/pretrained_model.keras)

### Load the Model Programmatically

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

model_path = hf_hub_download(
    repo_id="Raghava-Ram/brain-tumor-efficientnet",
    filename="pretrained_model.keras"
)

model = tf.keras.models.load_model(model_path)
```

## Model Details

- **Architecture:** EfficientNetB4 (Transfer Learning)
- **Input size:** 380×380 RGB
- **Classes:**
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- **Accuracy:** ~95%
- **Training Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle

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

## Running the Application

This project provides three different ways to run the web interface. Choose the one that best suits your needs:

### Option 1: Gradio App (Recommended for quick testing)
Gradio provides a beautiful, modern UI out-of-the-box with customized theme parsing.
```bash
python app.py
```
This will launch the app and provide a local URL (typically `http://127.0.0.1:7860`).

### Option 2: Streamlit App (Data-focused UI)
Streamlit executes as a straightforward dashboard.
```bash
streamlit run streamlit_app.py
```
This will open the application in your browser (typically `http://localhost:8501`).

### Option 3: FastAPI Backend + Custom HTML/JS Frontend (Full-stack approach)
If you want to run the model as a robust REST API with a glassmorphism frontend serving static assets.
```bash
python -m uvicorn main:app --reload
```
This will launch the backend and serve the application UI. Open your web browser and navigate to: **http://127.0.0.1:8000**
