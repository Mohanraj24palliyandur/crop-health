import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import io

def compute_ndvi(image_data):
    """
    Compute NDVI from uploaded image data
    Args:
        image_data: BytesIO object or file path
    """
    try:
        # Convert uploaded file to numpy array
        if isinstance(image_data, str):
            image = cv2.imread(image_data)
        else:
            # Convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not load image")

        # Extract channels
        red = image[:, :, 2].astype(float)  # Red channel
        nir = image[:, :, 0].astype(float)  # Near-Infrared channel

        # Avoid division by zero
        denominator = (nir + red)
        denominator[denominator == 0] = 1e-10

        # Calculate NDVI
        ndvi = (nir - red) / denominator

        # Create figure for plotting
        fig, ax = plt.subplots()
        im = ax.imshow(ndvi, cmap='RdYlGn')
        plt.colorbar(im)
        
        return fig, ndvi

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.title("NDVI Calculator")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        fig, ndvi = compute_ndvi(uploaded_file)
        
        if fig is not None:
            st.pyplot(fig)
            
            # Display NDVI statistics
            st.write("NDVI Statistics:")
            st.write(f"Mean NDVI: {np.mean(ndvi):.3f}")
            st.write(f"Max NDVI: {np.max(ndvi):.3f}")
            st.write(f"Min NDVI: {np.min(ndvi):.3f}")
    else:
        st.info("Please upload an image file")

if __name__ == "__main__":
    main()
