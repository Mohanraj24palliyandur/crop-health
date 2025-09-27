# Copy your CROP_DATA and CROP_MANAGEMENT dictionaries here
CROP_DATA = {
    "Rice (Paddy)": {
        "optimal_temp": "20-35°C",
        "water_needs": "High",
        "varieties": ["Basmati", "IR64", "Swarna", "MTU1010"],
        "diseases": {
            "Blast": {
                "symptoms": ["Spindle-shaped lesions", "White-gray spots", "Dark brown borders"],
                "solutions": ["Use resistant varieties", "Fungicide application", "Balanced fertilization"]
            },
            "Bacterial Blight": {
                "symptoms": ["Yellow-white lesions", "Leaf wilting", "Pale yellow leaves"],
                "solutions": ["Use disease-free seeds", "Crop rotation", "Copper-based bactericides"]
            }
        }
    },
    "Wheat": {
        "optimal_temp": "15-25°C",
        "water_needs": "Moderate",
        "varieties": ["HD2967", "PBW343", "WH542", "DBW187"],
        "diseases": {
            "Rust": {
                "symptoms": ["Orange-brown pustules", "Damaged leaf tissue"],
                "solutions": ["Early sowing", "Resistant varieties", "Fungicide treatment"]
            }
        }
    }
}

CROP_MANAGEMENT = {
    # ...existing CROP_MANAGEMENT data...
}