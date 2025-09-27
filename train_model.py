import tensorflow as tf
import deeplake
from pathlib import Path
import numpy as np

def create_model():
    """Create a simple CNN model for crop disease classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train model using PlantVillage dataset"""
    try:
        # Load dataset
        ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')
        
        # Convert to TensorFlow dataset
        train_ds = ds.tensorflow()
        
        # Create and train model
        model = create_model()
        
        history = model.fit(
            train_ds,
            epochs=10,
            validation_split=0.2
        )
        
        # Save model
        model.save('models/crop_disease_model.h5')
        print("✅ Model trained and saved successfully!")
        
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")

if __name__ == "__main__":
    train_model()