# Brain Tumor MRI Classifier

This repository contains a brain tumor MRI classification model and a Streamlit web application to test the model on new MRI images.

## Setup Instructions

### 1. Create an Anaconda Environment

It is recommended to use an Anaconda environment to manage dependencies. Open your Anaconda Prompt or terminal and run:

```bash
conda create -n brain-tumor-env python=3.10 -y
conda activate brain-tumor-env
```

### 2. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

Once the environment is set up and the packages are installed, make sure `Pretrained_model.keras` is in the same folder as `app.py`. Then, start the Streamlit application:

```bash
streamlit run app.py
```

This will launch the application in your default web browser, where you can upload MRI images to the classification model.

## Note on Pre-trained Model
The pre-trained model `Pretrained_model.keras` is ignored in Git due to its large size (exceeding GitHub's 100MB limit). You will need to keep this model locally in the project directory when running the app.
