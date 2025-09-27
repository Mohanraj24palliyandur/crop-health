import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define a simple NDVI computation function
def compute_ndvi(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Example: Assume NIR is in channel 0 and Red is in channel 2
        nir = image[:, :, 0].astype(float)
        red = image[:, :, 2].astype(float)
        ndvi = (nir - red) / (nir + red + 1e-6)
        return ndvi
    except Exception as e:
        print(f"Error computing NDVI: {str(e)}")
        return None

def load_models():
    try:
        # Load CNN model
        cnn_model = keras.models.load_model('cnn_model.h5')
        
        # Load LSTM model
        lstm_model = keras.models.load_model('lstm_model.h5')
        
        return cnn_model, lstm_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

def main():
    # Load the image and compute NDVI
    ndvi = compute_ndvi('hyperspectral_image.jpg')
    if ndvi is None:
        return
    
    # Load models
    cnn_model, lstm_model = load_models()
    if cnn_model is None or lstm_model is None:
        return
        
    try:
        # Make CNN prediction
        cnn_prediction = cnn_model.predict(np.expand_dims(ndvi, axis=0))
        
        # Example sensor data (replace with actual data)
        sensor_time_series = np.random.rand(1, 10, 5)  # (batch, timesteps, features)
        
        # Make LSTM prediction
        pest_risk_prediction = lstm_model.predict(sensor_time_series)
        
        # Trigger alerts if pest risk is above threshold
        if pest_risk_prediction > 0.7:
            print("Alert: High pest risk detected!")
            
    except Exception as e:
        print(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()
