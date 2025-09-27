import tensorflow as tf
from keras import layers, models  # Updated import
import numpy as np

def build_cnn_model(input_shape):
    """Build and compile CNN model for crop health classification"""
    try:
        model = models.Sequential([
            layers.Input(shape=input_shape),  # Explicit input layer
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        return model
    
    except Exception as e:
        print(f"Error building model: {str(e)}")
        return None

def main():
    # Example input shape (height, width, channels)
    input_shape = (256, 256, 3)  # Hyperspectral image dimensions
    
    # Build model
    cnn_model = build_cnn_model(input_shape)
    if cnn_model is not None:
        cnn_model.summary()

if __name__ == "__main__":
    main()
