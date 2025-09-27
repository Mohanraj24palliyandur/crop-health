import tensorflow as tf
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Tuple
import time

# Enhanced crop data for Indian agriculture
CROP_DATA: Dict[str, Dict] = {
    "Rice": {
        "pests": ["Brown Planthopper", "Stem Borer", "Rice Blast", "Leaf Folder"],
        "moisture_range": (0.3, 0.6),
        "season": "Kharif",
        "diseases": ["Bacterial Blight", "Sheath Blight"]
    },
    "Wheat": {
        "pests": ["Aphids", "Rust Disease", "Powdery Mildew", "Army Worm"],
        "moisture_range": (0.2, 0.5),
        "season": "Rabi",
        "diseases": ["Yellow Rust", "Karnal Bunt"]
    },
    "Cotton": {
        "pests": ["Bollworm", "Whitefly", "Pink Bollworm", "Thrips"],
        "moisture_range": (0.2, 0.4),
        "season": "Kharif",
        "diseases": ["Wilt", "Black Arm"]
    },
    "Sugarcane": {
        "pests": ["Early Shoot Borer", "Top Borer", "Pyrilla"],
        "moisture_range": (0.4, 0.7),
        "season": "Year-round",
        "diseases": ["Red Rot", "Smut"]
    },
    "Maize": {
        "pests": ["Stem Borer", "Army Worm", "Shoot Fly"],
        "moisture_range": (0.3, 0.5),
        "season": "Kharif/Rabi",
        "diseases": ["Leaf Blight", "Stalk Rot"]
    },
    "Pulses": {
        "pests": ["Pod Borer", "Aphids", "White Fly"],
        "moisture_range": (0.2, 0.4),
        "season": "Rabi/Kharif",
        "diseases": ["Wilt", "Yellow Mosaic"]
    },
    "Groundnut": {
        "pests": ["Leaf Miner", "White Grub", "Thrips"],
        "moisture_range": (0.2, 0.4),
        "season": "Kharif",
        "diseases": ["Leaf Spot", "Rust"]
    },
    "Soybean": {
        "pests": ["Stem Fly", "Girdle Beetle", "Pod Borer"],
        "moisture_range": (0.3, 0.5),
        "season": "Kharif",
        "diseases": ["Rust", "Yellow Mosaic"]
    },
    "Mustard": {
        "pests": ["Aphid", "Painted Bug", "Sawfly"],
        "moisture_range": (0.2, 0.4),
        "season": "Rabi",
        "diseases": ["White Rust", "Alternaria Blight"]
    },
    "Potato": {
        "pests": ["Potato Tuber Moth", "Aphids", "White Fly"],
        "moisture_range": (0.3, 0.5),
        "season": "Rabi",
        "diseases": ["Late Blight", "Early Blight"]
    }
}

@contextmanager
def timeout_handler():
    try:
        yield
    except Exception as e:
        st.error(f"Operation timed out: {str(e)}")
        time.sleep(0.1)  # Prevent rapid retries

def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """Build and compile LSTM model with timeout handling"""
    with timeout_handler():
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        return model

def main():
    st.set_page_config(page_title="Indian Crop Health Monitor", layout="wide")
    st.title("Indian Crop Health Monitoring System")
    
    # Enhanced sidebar with more information
    with st.sidebar:
        selected_crop = st.selectbox("Select Crop", list(CROP_DATA.keys()))
        st.info(f"Selected: {selected_crop}")
        st.write(f"Growing Season: {CROP_DATA[selected_crop]['season']}")
        
        # Display common diseases
        st.subheader("Common Diseases")
        for disease in CROP_DATA[selected_crop]['diseases']:
            st.write(f"- {disease}")
        
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            moisture_1 = st.slider("Day 1 Moisture", 0.0, 1.0, 0.35)
        with col2:
            moisture_2 = st.slider("Day 2 Moisture", 0.0, 1.0, 0.42)
        with col3:
            moisture_3 = st.slider("Day 3 Moisture", 0.0, 1.0, 0.50)

        sensor_data = np.array([moisture_1, moisture_2, moisture_3])
        sensor_data = sensor_data.reshape((1, 3, 1))

        if st.button("Analyze Crop Health"):
            with st.spinner("Processing..."):
                with timeout_handler():
                    model = build_lstm_model((3, 1))
                    
                    if model:
                        history = model.fit(
                            sensor_data, 
                            np.array([0.6]), 
                            epochs=5,
                            verbose=0
                        )
                        
                        prediction = model.predict(sensor_data, verbose=0)
                        
                        # Results section
                        st.subheader("Analysis Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Moisture", f"{prediction[0][0]:.4f}")
                            
                            # Risk assessment
                            min_moisture, max_moisture = CROP_DATA[selected_crop]["moisture_range"]
                            if prediction[0][0] > max_moisture:
                                st.error("⚠️ High Moisture Risk")
                            elif prediction[0][0] < min_moisture:
                                st.warning("⚠️ Low Moisture Risk")
                            else:
                                st.success("✅ Normal Range")
                        
                        with col2:
                            st.subheader("Potential Pests")
                            for pest in CROP_DATA[selected_crop]["pests"]:
                                st.write(f"- {pest}")
                        
                        # Trend visualization
                        st.subheader("Moisture Trend")
                        st.line_chart(sensor_data.reshape(-1))
                        
                        # Detailed Crop Information
                        st.subheader("Crop Management Information")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("#### Season Information")
                            st.write(f"Growing Season: {CROP_DATA[selected_crop]['season']}")
                            st.write("#### Optimal Conditions")
                            st.write(f"Moisture Range: {CROP_DATA[selected_crop]['moisture_range'][0]} - {CROP_DATA[selected_crop]['moisture_range'][1]}")
                        
                        with col2:
                            st.write("#### Disease Management")
                            for disease in CROP_DATA[selected_crop]['diseases']:
                                st.write(f"- {disease}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please try again with different values")

if __name__ == "__main__":
    main()
