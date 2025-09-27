import os
import requests
from PIL import Image
import io
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

class SentinelAPI:
    def __init__(self):
        self.instance_id = os.getenv("SENTINEL_INSTANCE_ID")
        self.base_url = f"https://services.sentinel-hub.com/ogc/wms/{self.instance_id}"
        
        if not self.instance_id:
            raise ValueError("Sentinel Hub Instance ID not found in environment variables")

    def get_ndvi_image(self, bbox, date=None):
        """
        Fetch NDVI data for a given bounding box and date
        Args:
            bbox (str): Bounding box coordinates "min_lon,min_lat,max_lon,max_lat"
            date (str): Date in YYYY-MM-DD format
        """
        try:
            if date is None:
                date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            params = {
                "SERVICE": "WMS",
                "REQUEST": "GetMap",
                "LAYERS": "NDVI",
                "BBOX": bbox,
                "CRS": "EPSG:4326",
                "WIDTH": "512",
                "HEIGHT": "512",
                "FORMAT": "image/png",
                "TIME": date
            }

            with st.spinner("Fetching satellite data..."):
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    return Image.open(io.BytesIO(response.content))
                else:
                    st.error(f"API Error: {response.status_code}")
                    return None

        except Exception as e:
            st.error(f"Failed to fetch NDVI data: {str(e)}")
            return None

    @staticmethod
    def create_bbox(lat, lon, radius_km=1):
        """Create a bounding box around a point"""
        # Approximate 1km at equator
        deg_per_km = 0.009
        offset = deg_per_km * radius_km
        return f"{lon-offset},{lat-offset},{lon+offset},{lat+offset}"