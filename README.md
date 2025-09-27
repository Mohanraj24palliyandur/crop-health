
# AI Crop Health Monitoring System

## Overview
This project leverages **Artificial Intelligence (AI)** and **Machine Learning (ML)** to monitor crop health using image analysis. The system can **detect crop diseases early**, helping farmers reduce losses and improve yield.

---

## Features
- Upload images of crops via a web interface.
- Detect whether a crop is **healthy** or **diseased**.
- Provides **confidence scores** for predictions.
- Built with **Python, Streamlit, and ML models**.

---

Step-by-Step Process
1️⃣ Project Overview

Purpose: To help farmers monitor crop health using AI and image analysis.

Goal: Detect crop diseases early to prevent crop loss.

2️⃣ Data Collection

Collect images of crops (healthy vs diseased).

Example sources: Kaggle datasets, research datasets, or self-captured images.

Organize the dataset into folders:

Dataset/
    Healthy/
    Diseased/

3️⃣ Data Preprocessing

Resize images to a standard size (e.g., 224x224).

Normalize pixel values to a 0–1 scale.

Label the images (e.g., 0 = Healthy, 1 = Diseased).

Split the dataset into:

Training set (e.g., 80%)

Testing set (e.g., 20%)

4️⃣ Model Development

Choose a Machine Learning / Deep Learning model:

CNN (Convolutional Neural Network) for image classification.

Build the model using TensorFlow / Keras.

Compile with:

Loss: categorical_crossentropy

Optimizer: adam

Metrics: accuracy

5️⃣ Model Training

Train the model on the training dataset.

Validate using the testing dataset.

Save the trained model (e.g., train_model.h5) for future use.

6️⃣ Building the Web App

Use Streamlit to create a simple web interface.

Features:

Upload crop images.

Predict if the crop is healthy or diseased.

Show confidence scores for the prediction.

7️⃣ Integration & Testing

Integrate the trained model with the Streamlit app.

Test with new images to check accuracy.

Refine preprocessing or model parameters if results are poor.

8️⃣ Deployment

Run locally:

streamlit run app.py


Optional: Deploy online using Streamlit Cloud or Heroku.

9️⃣ Optional Enhancements

Add disease-specific suggestions for farmers.

Support multiple crop types.

Include visualizations of crop health trends.

10️⃣ Usage Instructions (for README)

Clone the repository:

git clone https://github.com/Mohanraj24palliyandur/crop-health.git


Navigate to project folder:

cd crop-health


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py
