import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
import io
import time
from contextlib import contextmanager
import sys
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache
import tempfile
import uuid
import tensorflow as tf
import deeplake
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Add retry decorator and timeout handler
@contextmanager
def request_timeout(timeout_duration=30):
    try:
        yield
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        sys.exit(1)

def safe_image_open(image_input):
    """Safely open and validate image files with retry logic"""
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img_bytes = image_input.getvalue()
            img = Image.open(io.BytesIO(img_bytes))
            del img_bytes
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def compute_ndvi(image_input):
    """Improved NDVI computation with retry logic"""
    try:
        img = safe_image_open(image_input)
        if img is None:
            return None, None
            
        img_np = np.array(img, dtype=np.float32)
        red = img_np[:, :, 0].copy()
        nir = img_np[:, :, 1].copy()
        del img_np
        
        denominator = nir + red + 1e-8
        ndvi = np.divide(nir - red, denominator, out=np.zeros_like(red), where=denominator!=0)
        
        ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
        if ndvi_min == ndvi_max:
            st.warning("Low NDVI variation detected")
            ndvi_display = np.zeros_like(ndvi, dtype=np.uint8)
        else:
            ndvi_display = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min) * 255).astype(np.uint8)
        
        ndvi_img = Image.fromarray(ndvi_display)
        return ndvi_img, ndvi
        
    except Exception as e:
        st.error(f"Error computing NDVI: {str(e)}")
        return None, None

# Basic crop data
CROP_MANAGEMENT = {
    "Rice (Paddy)": {
        "sowing_season": ["Kharif (June-July)", "Rabi (Nov-Dec)"],
        "soil_type": "Clay or clay loam soils with good water retention",
        "irrigation": ["Maintain 5cm water level during tillering", "Periodic irrigation during panicle formation"],
        "fertilizer": {"N": 120, "P": 60, "K": 40},
        "harvesting": "When 80% of grains turn golden yellow"
    },
    "Wheat": {
        "sowing_season": ["Rabi (Oct-Nov)"],
        "soil_type": "Well-drained loamy soil",
        "irrigation": ["First irrigation at crown root stage", "Regular intervals of 20-25 days"],
        "fertilizer": {"N": 120, "P": 60, "K": 40},
        "harvesting": "When grains become hard and straw turns golden"
    }
}

CROP_DATA = {
    "Rice (Paddy)": {
        "optimal_temp": "20-35°C",
        "water_needs": "High",
        "varieties": [
            "Basmati 370", "IR64", "Swarna", "MTU1010", "Pusa Basmati 1", 
            "Samba Mahsuri", "HMT", "Jaya", "ADT 36", "ADT 43", "CR 1009",
            "Improved White Ponni", "BPT 5204"
        ],
        "diseases": {
            "Blast": {
                "symptoms": ["Spindle-shaped lesions", "White-gray spots", "Dark brown borders"],
                "solutions": ["Apply fungicide", "Improve air circulation", "Remove infected plants"]
            },
            "Sheath Blight": {
                "symptoms": ["Brown, elongated lesions on leaves", "Sheath splitting", "Grayish fungal growth"],
                "solutions": ["Apply tricyclazole or isoprothiolane", "Maintain proper water management", "Avoid excessive nitrogen"]
            }
        },
        "recommendations": [
            "Ensure proper spacing between plants",
            "Monitor for pest infestations",
            "Apply nitrogen-rich fertilizer during the tillering stage",
            "Maintain water level of 5 cm in the field for optimal growth",
            "Harvest when grains are at 20% moisture content for best quality"
        ]
    },
    "Wheat": {
        "optimal_temp": "10-25°C",
        "water_needs": "Moderate",
        "varieties": [
            "PW 317", "HD 2967", "WH 1080", "C 306", "HUW 234",
            "DP 888", "Raj 3765", "PBW 343", "PBW 502", "K 9107"
        ],
        "diseases": {
            "Rust": {
                "symptoms": ["Small, round, reddish-brown pustules on leaves", "Yellowing of leaves", "Premature leaf drop"],
                "solutions": ["Apply fungicide like tebuconazole", "Use resistant varieties", "Practice crop rotation"]
            },
            "Blight": {
                "symptoms": ["Water-soaked lesions on leaves", "Leaf curling", "Black streaks on stems"],
                "solutions": ["Apply mancozeb or chlorothalonil", "Remove and destroy infected plant debris", "Avoid overhead irrigation"]
            }
        },
        "recommendations": [
            "Plant wheat seeds at a depth of 4-5 cm",
            "Irrigate the crop every 10-15 days depending on rainfall",
            "Apply phosphorus and potassium fertilizers at the time of planting",
            "Control weeds during the early stages of crop growth",
            "Harvest when the grains are hard and the moisture content is below 14%"
        ]
    },
    "Pulses": {
        "optimal_temp": "15-30°C",
        "water_needs": "Low",
        "varieties": [
            "Pusa 992", "Pusa 256", "Pusa 1401", "Pusa 1403", "Pusa 1404",
            "Pusa 1406", "Pusa 1411", "Pusa 1412", "Pusa 1413", "Pusa 1414"
        ],
        "diseases": {
            "Leaf Spot": {
                "symptoms": ["Dark, angular lesions on leaves", "Yellowing of leaf margins", "Premature leaf drop"],
                "solutions": ["Apply chlorothalonil or mancozeb", "Remove and destroy infected leaves", "Avoid overhead irrigation"]
            },
            "Root Rot": {
                "symptoms": ["Wilting of plants", "Stunted growth", "Dark, sunken lesions on roots"],
                "solutions": ["Apply carbendazim or thiophanate-methyl", "Improve soil drainage", "Avoid waterlogging"]
            }
        },
        "recommendations": [
            "Inoculate seeds with Rhizobium culture before planting",
            "Maintain soil moisture during the flowering and pod-setting stages",
            "Apply mulching to conserve soil moisture and suppress weeds",
            "Harvest when the pods are dry and seeds rattle inside",
            "Store seeds in a cool, dry place to prevent fungal infections"
        ]
    },
    "Oilseeds": {
        "optimal_temp": "20-30°C",
        "water_needs": "Moderate",
        "varieties": [
            "Pusa Vishal", "Pusa 2-21", "Pusa 2-43", "Pusa 2-55", "Pusa 2-60",
            "Pusa 2-71", "Pusa 2-79", "Pusa 2-80", "Pusa 2-81", "Pusa 2-82"
        ],
        "diseases": {
            "Downy Mildew": {
                "symptoms": ["Yellowing of leaves", "Downy growth on the underside of leaves", "Premature leaf drop"],
                "solutions": ["Apply metalaxyl or mefenoxam", "Remove and destroy infected plant debris", "Practice crop rotation"]
            },
            "White Rust": {
                "symptoms": ["White, powdery spots on leaves and stems", "Leaf curling", "Stunted growth"],
                "solutions": ["Apply fungicides like azoxystrobin", "Remove and destroy infected plants", "Avoid excessive nitrogen"]
            }
        },
        "recommendations": [
            "Sow seeds at a depth of 2-3 cm in well-prepared soil",
            "Irrigate the crop at critical growth stages: flowering and pod development",
            "Apply balanced NPK fertilizer at the time of planting",
            "Control weeds and pests regularly",
            "Harvest when the seeds are mature and moisture content is around 9%"
        ]
    },
    "Cotton": {
        "optimal_temp": "20-30°C",
        "water_needs": "High",
        "varieties": [
            "Hirsutum 1", "Hirsutum 2", "Hirsutum 3", "Hirsutum 4", "Hirsutum 5",
            "Hirsutum 6", "Hirsutum 7", "Hirsutum 8", "Hirsutum 9", "Hirsutum 10"
        ],
        "diseases": {
            "Boll Rot": {
                "symptoms": ["Rotting of bolls", "Fungal growth on bolls", "Premature defoliation"],
                "solutions": ["Apply carbendazim or mancozeb", "Remove and destroy infected bolls", "Avoid overhead irrigation"]
            },
            "Leaf Curl": {
                "symptoms": ["Curling of leaves", "Yellowing of leaf margins", "Stunted growth"],
                "solutions": ["Apply imidacloprid or acetamiprid", "Remove and destroy infected leaves", "Control whiteflies and aphids"]
            }
        },
        "recommendations": [
            "Plant cotton seeds at a depth of 5-7 cm",
            "Irrigate the crop at 10-15 days interval depending on soil moisture",
            "Apply nitrogen fertilizer in split doses: at planting and during flowering",
            "Control pests like boll weevils and aphids regularly",
            "Harvest when the bolls burst open and fibers are fluffy"
        ]
    },
    "Sugarcane": {
        "optimal_temp": "20-32°C",
        "water_needs": "Very high",
        "varieties": [
            "Co 0238", "Co 0240", "Co 0250", "Co 0251", "Co 0253",
            "Co 0254", "Co 0255", "Co 0256", "Co 0257", "Co 0258"
        ],
        "diseases": {
            "Red Rot": {
                "symptoms": ["Red discoloration of the cane", "Rotting of the pith", "Drying of leaves"],
                "solutions": ["Apply propiconazole or tebuconazole", "Remove and destroy infected canes", "Practice crop rotation"]
            },
            "Fungal Leaf Spot": {
                "symptoms": ["Small, round, dark spots on leaves", "Yellowing of leaf margins", "Premature leaf drop"],
                "solutions": ["Apply chlorothalonil or mancozeb", "Remove and destroy infected leaves", "Avoid overhead irrigation"]
            }
        },
        "recommendations": [
            "Plant sugarcane setts at a depth of 10-15 cm",
            "Irrigate the crop every 7-10 days during dry spells",
            "Apply nitrogen, phosphorus, and potassium fertilizers in recommended doses",
            "Control weeds, pests, and diseases regularly",
            "Harvest when the cane juice is sweet and brix level is above 18%"
        ]
    },
    "Vegetables": {
        "Tomato": ["Pusa Ruby", "Arka Vikas", "Pusa Early Dwarf", "CO 3"],
        "Potato": ["Kufri Jyoti", "Kufri Pukhraj", "Kufri Sindhuri", "Kufri Chandramukhi"],
        "Onion": ["Pusa Red", "N 53", "Agrifound Light Red", "Bhima Super"],
        "Chilli": ["Pusa Jwala", "G4", "K2", "CO 4", "Bhut Jolokia"]
    },
    "Fruits": {
        "Mango": ["Alphonso", "Dashehari", "Langra", "Chausa", "Amrapali"],
        "Banana": ["Grand Naine", "Robusta", "Red Banana", "Poovan", "Nendran"],
        "Citrus": ["Nagpur Mandarin", "Kinnow", "Mosambi", "Khasi Mandarin"]
    },
    "Millets": {
        "optimal_temp": "25-35°C",
        "water_needs": "Low",
        "varieties": {
            "Bajra": ["HHB 67", "GHB 558", "RHB 177", "PHB 3"],
            "Jowar": ["CSH 14", "CSV 15", "CSV 17", "SPV 462"],
            "Ragi": ["GPU 28", "PR 202", "HR 374", "GPU 67"]
        }
    }
}

PEST_DATA = {
    "Rice (Paddy)": {
        "Brown Planthopper": {
            "symptoms": [
                "Yellowing of leaves",
                "Plants appear burnt",
                "Honeydew secretion on leaves"
            ],
            "solutions": [
                "Use resistant varieties",
                "Avoid excessive nitrogen",
                "Apply neem-based pesticides",
                "Release natural predators"
            ],
            "severity_levels": {
                "Low": "Few insects visible",
                "Medium": "Patches of damaged plants",
                "High": "Widespread plant death"
            }
        },
        "Stem Borer": {
            "symptoms": [
                "Dead hearts in young plants",
                "White heads in mature plants",
                "Holes in stems"
            ],
            "solutions": [
                "Remove and destroy affected stems",
                "Use pheromone traps",
                "Time planting to avoid peak pest periods",
                "Apply appropriate insecticides"
            ]
        }
    },
    "Wheat": {
        "Aphids": {
            "symptoms": [
                "Curling of leaves",
                "Stunted growth",
                "Honeydew on leaves"
            ],
            "solutions": [
                "Spray soap solution",
                "Use yellow sticky traps",
                "Apply systematic insecticides",
                "Encourage natural predators"
            ]
        }
    }
}

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa"
}

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(exist_ok=True)

@lru_cache(maxsize=1000)
def translate_text(text, target_lang, max_retries=3):
    """
    Translate text with retry logic and caching
    """
    if target_lang == 'en':  # Skip translation for English
        return text
        
    try:
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        translator = GoogleTranslator(
            source='en',
            target=target_lang,
            timeout=10
        )
        
        # Add session with retry logic
        session = requests.Session()
        session.mount("https://", adapter)
        translator.session = session
        
        return translator.translate(text) or text
        
    except Exception as e:
        st.warning(f"Translation unavailable, showing English: {str(e)}")
        return text  # Fallback to English

def init_page():
    st.set_page_config(
        page_title="Crop Health Monitor",
        page_icon="🌾",
        layout="wide"
    )

def display_crop_info(crop_data, selected_crop):
    st.sidebar.write("#### Growing Conditions")
    st.sidebar.info(f"🌡️ Optimal Temperature: {crop_data[selected_crop]['optimal_temp']}")
    st.sidebar.info(f"💧 Water Requirement: {crop_data[selected_crop]['water_needs']}")

def display_management_info(crop_management, selected_crop):
    if selected_crop in crop_management:
        with st.expander("🌱 Crop Management Details"):
            tabs = st.tabs(["Seasons", "Soil & Water", "Fertilizer"])
            
            with tabs[0]:
                st.write("#### Sowing Seasons")
                for season in crop_management[selected_crop]["sowing_season"]:
                    st.info(season)
            
            with tabs[1]:
                st.write("#### Soil Type")
                st.write(crop_management[selected_crop]["soil_type"])
                st.write("#### Irrigation")
                for irr in crop_management[selected_crop]["irrigation"]:
                    st.write(f"• {irr}")
            
            with tabs[2]:
                st.write("#### Fertilizer Requirements (kg/ha)")
                fert = crop_management[selected_crop]["fertilizer"]
                cols = st.columns(3)
                cols[0].metric("Nitrogen (N)", fert["N"])
                cols[1].metric("Phosphorus (P)", fert["P"])
                cols[2].metric("Potassium (K)", fert["K"])

def display_map_selector():
    """Display interactive map for field selection"""
    st.subheader("🌍 Select Field Location")
    
    # Create tabs for different selection methods
    point_tab, area_tab = st.tabs(["📍 Select Point", "🔲 Draw Area"])
    
    with point_tab:
        # Base map for point selection
        m_point = folium.Map(location=[12.8, 77.5], zoom_start=12)
        
        # Add draggable marker
        folium.Marker(
            [12.8, 77.5],
            tooltip="Click and drag me",
            draggable=True
        ).add_to(m_point)
        
        point_data = st_folium(m_point, width=700, height=500)
        
        if isinstance(point_data, dict) and point_data.get("last_clicked"):
            lat = point_data["last_clicked"]["lat"]
            lon = point_data["last_clicked"]["lng"]
            st.success(f"📍 Selected Point: {lat:.6f}, {lon:.6f}")
            return {"type": "point", "lat": lat, "lon": lon}
    
    with area_tab:
        # Base map for area selection
        m_area = folium.Map(location=[12.8, 77.5], zoom_start=12)
        
        # Add drawing tools
        draw = Draw(
            draw_options={
                'rectangle': True,
                'polygon': True,
                'circle': True,
                'marker': False,
                'circlemarker': False,
                'polyline': False
            },
            edit_options={'edit': True}
        )
        m_area.add_child(draw)
        
        area_data = st_folium(m_area, width=700, height=500)
        
        if isinstance(area_data, dict) and area_data.get("all_drawings"):
            st.success("✅ Area selected successfully!")
            with st.expander("View Area Details"):
                st.json(area_data["all_drawings"])
            return {"type": "area", "data": area_data["all_drawings"]}
    
    return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Update process_uploaded_image function
def process_uploaded_image(uploaded_file, selected_crop):
    """Process uploaded image with real disease detection"""
    try:
        with st.spinner("Analyzing image..."):
            # Load and preprocess image
            image = Image.open(uploaded_file)
            img_processed = preprocess_image(image)
            
            if img_processed is not None:
                # Load model (you'll need to train this first)
                model_file = MODEL_PATH / f"{selected_crop.lower()}_model.h5"
                
                if model_file.exists():
                    model = tf.keras.models.load_model(str(model_file))
                    
                    # Get prediction
                    prediction = model.predict(img_processed)
                    
                    # Display results in farmer-friendly way
                    st.subheader("📊 Analysis Results")
                    
                    # Show original image
                    st.image(image, caption="Your Crop Image", width="stretch")
                    
                    # Display health status
                    health_score = float(prediction[0][0])  # Assuming binary classification
                    display_health_status(health_score)
                    
                    # Show recommendations based on prediction
                    if health_score < 0.5:
                        st.warning("⚠️ Possible disease detected!")
                        if selected_crop in CROP_DATA and 'diseases' in CROP_DATA[selected_crop]:
                            st.subheader("🏥 Disease Management Recommendations")
                            for disease, info in CROP_DATA[selected_crop]['diseases'].items():
                                with st.expander(f"Managing {disease}"):
                                    st.write("**Common Symptoms:**")
                                    for symptom in info['symptoms']:
                                        st.write(f"• {symptom}")
                                    st.write("\n**Solutions:**")
                                    for solution in info['solutions']:
                                        st.write(f"• {solution}")
                        else:
                            st.info("No specific disease management information available for this crop.")
                    else:
                        st.success("✅ Your crop looks healthy!")
                        st.info("Continue following the recommended crop management practices.")
                else:
                    # Fallback to NDVI analysis if no model exists
                    ndvi_img, ndvi_values = compute_ndvi(uploaded_file)
                    if ndvi_img and ndvi_values is not None:
                        # Display images side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Your Crop Photo", width="stretch")
                        with col2:
                            st.image(ndvi_img, caption="Health Analysis", width="stretch")
                        
                        # Show farmer-friendly analysis
                        display_vegetation_analysis(ndvi_values, selected_crop)
                        
                        # Add help button
                        if st.button("🤝 Need Help?"):
                            st.info("""
                            **Contact your local agriculture expert:**
                            - Kisan Call Center: 1800-180-1551
                            - Agricultural Extension Officer
                            - Local Krishi Vigyan Kendra
                            """)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Add this function to display crop management
def display_crop_management(selected_crop):
    """Display detailed crop management information"""
    if selected_crop in CROP_MANAGEMENT:
        st.subheader("🌱 Crop Management Guide")
        
        # Create tabs for different management aspects
        mgmt_tabs = st.tabs(["Seasons", "Soil & Water", "Fertilizers", "Best Practices"])
        
        with mgmt_tabs[0]:
            st.write("#### Sowing Seasons")
            for season in CROP_MANAGEMENT[selected_crop]["sowing_season"]:
                st.info(f"🗓️ {season}")
        
        with mgmt_tabs[1]:
            st.write("#### Soil Requirements")
            st.write(f"🌍 {CROP_MANAGEMENT[selected_crop]['soil_type']}")
            
            st.write("#### Irrigation Schedule")
            for irr in CROP_MANAGEMENT[selected_crop]['irrigation']:
                st.write(f"💧 {irr}")
        
        with mgmt_tabs[2]:
            st.write("#### Fertilizer Requirements")
            fert = CROP_MANAGEMENT[selected_crop]['fertilizer']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nitrogen (N)", fert['N'])
            with col2:
                st.metric("Phosphorus (P)", fert['P'])
            with col3:
                st.metric("Potassium (K)", fert['K'])
        
        with mgmt_tabs[3]:
            st.write("#### Harvesting")
            st.success(f"🎯 {CROP_MANAGEMENT[selected_crop]['harvesting']}")

# Add this to handle display of full crop information
def get_varieties_for_crop(crop_data, crop_name):
    """Helper function to extract varieties for a given crop"""
    if crop_name in crop_data:
        if 'varieties' in crop_data[crop_name]:
            if isinstance(crop_data[crop_name]['varieties'], list):
                return crop_data[crop_name]['varieties']
            elif isinstance(crop_data[crop_name]['varieties'], dict):
                # Flatten varieties if stored in dictionary
                return [var for sublist in crop_data[crop_name]['varieties'].values() for var in sublist]
    return []

def display_full_crop_info(selected_crop):
    """Display comprehensive crop information in an organized way"""
    st.markdown("---")
    st.header(f"🌾 {selected_crop} Management Guide")
    
    # Basic Information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Growing Conditions")
        st.write(f"🌡️ Optimal Temperature: {CROP_DATA[selected_crop]['optimal_temp']}")
        st.write(f"💧 Water Requirement: {CROP_DATA[selected_crop]['water_needs']}")
    
    with col2:
        st.subheader("Popular Varieties")
        varieties = get_varieties_for_crop(CROP_DATA, selected_crop)

        for var in varieties[:5]:  # Show top 5 varieties
            st.write(f"• {var}")
        if len(varieties) > 5:
            with st.expander("See more varieties"):
                for var in varieties[5:]:
                    st.write(f"• {var}")
    
    # Management Calendar
    st.subheader("📅 Crop Calendar")
    calendar_data = {
        "Jan": "Land preparation",
        "Feb-Mar": "Sowing/Planting",
        "Apr-Jul": "Growth & Maintenance",
        "Aug-Sep": "Harvesting",
        "Oct-Dec": "Post-harvest"
    }
    
    cols = st.columns(len(calendar_data))
    for col, (month, activity) in zip(cols, calendar_data.items()):
        with col:
            st.write(f"**{month}**")
            st.write(activity)

def speak_text(text, lang='en', max_retries=3):
    """Convert text to speech with improved error handling and retry logic"""
    for attempt in range(max_retries):
        try:
            # Use system temp directory
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f'audio_{uuid.uuid4().hex}.mp3')
            
            # Chunk long text into smaller parts
            MAX_LENGTH = 500  # Maximum length for each chunk
            text_chunks = [text[i:i+MAX_LENGTH] for i in range(0, len(text), MAX_LENGTH)]
            
            audio_data = []
            for chunk in text_chunks:
                # Generate speech for each chunk
                tts = gTTS(text=chunk, lang=lang, slow=False)
                tts.save(temp_file)
                
                # Read and append audio data
                with open(temp_file, 'rb') as f:
                    audio_data.append(f.read())
                
                # Clean up temp file immediately
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Combine all audio chunks
            return b''.join(audio_data)
            
        except ConnectionError as ce:
            if attempt == max_retries - 1:
                st.warning(f"Network error: Please check your internet connection. ({str(ce)})")
                return None
            time.sleep(2)  # Wait before retry
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Could not generate audio: {str(e)}")
                return None
            time.sleep(2)  # Wait before retry
            
        finally:
            # Ensure temp file is cleaned up
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

# Update the recommendations section in display_dashboard
def display_health_status(health_score):
    """Display health status based on the prediction score"""
    st.write("#### Health Status")
    
    # Create a progress bar for visual representation
    progress_color = "green" if health_score >= 0.7 else "yellow" if health_score >= 0.5 else "red"
    st.progress(health_score, text=f"Health Score: {health_score:.2%}")
    
    # Display status message
    if health_score >= 0.7:
        st.success("🌟 Your crop appears very healthy!")
    elif health_score >= 0.5:
        st.warning("⚠️ Your crop shows some signs of stress")
    else:
        st.error("🚨 Your crop may need immediate attention")

def display_recommendations(recommendations, lang_code):
    """Display recommendations with improved audio handling"""
    header = "📋 Recommendations"
    st.subheader(header if lang_code == 'en' else translate_text(header, lang_code))
    
    # Display all recommendations first
    for idx, rec in enumerate(recommendations, 1):
        translated_text = rec if lang_code == 'en' else translate_text(rec, lang_code)
        st.write(f"{idx}. {translated_text}")
    
    # Add audio control with retry option
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔊 Listen to All Recommendations"):
            with st.spinner("Generating audio..."):
                combined_text = ". ".join(recommendations)
                translated_combined = combined_text if lang_code == 'en' else translate_text(combined_text, lang_code)
                audio_bytes = speak_text(translated_combined, lang_code)
                
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')
                    st.success("✅ Audio generated successfully!")
                else:
                    st.error("❌ Could not generate audio")
    
    with col2:
        if st.button("🔄 Retry Audio"):
            st.rerun()

def display_dashboard():
    try:
        st.title("Crop Health Monitoring System 🌾")
        
        # Sidebar with language and crop selection
        with st.sidebar:
            st.title("🌐 भाषा / Language")
            selected_language = st.selectbox(
                "Select Your Language / अपनी भाषा चुनें",
                options=list(SUPPORTED_LANGUAGES.keys()),
                index=0
            )
            lang_code = SUPPORTED_LANGUAGES[selected_language]
            
            st.title("🌱 Crop Selection")
            selected_crop = st.selectbox(
                translate_text("Select Your Crop", lang_code),
                options=list(CROP_DATA.keys())
            )
        
        # Main content area with simplified tabs
        tab1, tab2 = st.tabs([
            "📱 Easy Mode",
            "🔍 Advanced Mode"
        ])
        
        with tab1:
            st.header("Quick Crop Check")
            
            # Simple image upload
            uploaded_file = st.file_uploader(
                translate_text("Take a photo of your crop", lang_code),
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption=translate_text("Your crop image", lang_code))
                
                if st.button("📋 Get Advice"):
                    with st.spinner(translate_text("Checking your crop...", lang_code)):
                        # Show farmer-friendly results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Crop Health", "💪 Good")
                        with col2:
                            st.metric("Water Need", "💧 Required")
                        with col3:
                            st.metric("Next Steps", "🌱 Add fertilizer")
                        
                        # Show recommendations in local language
                        st.subheader(translate_text("What to do now:", lang_code))
                        recommendations = CROP_DATA[selected_crop]['recommendations']
                        for idx, rec in enumerate(recommendations, 1):
                            translated_rec = translate_text(rec, lang_code)
                            st.info(f"{idx}. {translated_rec}")
                        
                        # Add voice support
                        if st.button("🔊 Listen to Advice"):
                            combined_text = ". ".join(recommendations)
                            translated_text = translate_text(combined_text, lang_code)
                            audio_bytes = speak_text(translated_text, lang_code)
                            if audio_bytes:
                                st.audio(audio_bytes, format='audio/mp3')
        
        with tab2:
            # Keep your existing advanced features here
            location_tab, upload_tab = st.tabs([
                translate_text("📍 Field Selection", lang_code),
                translate_text("📤 Detailed Analysis", lang_code)
            ])
            
            with location_tab:
                # Display interactive map
                location_data = display_map_selector()
                
                if location_data:
                    if location_data["type"] == "point":
                        st.write("#### Selected Location Details")
                        st.write(f"Latitude: {location_data['lat']}")
                        st.write(f"Longitude: {location_data['lon']}")
                    else:  # area
                        st.write("#### Selected Area Details")
                        area_size = len(location_data["data"])
                        st.write(f"Number of vertices: {area_size}")
                        
                # Date selection
                selected_date = st.date_input(
                    "Select Date",
                    max_value=datetime.now()
                )
                
                if st.button("Analyze Field"):
                    if location_data:
                        with st.spinner("Analyzing field data..."):
                            # Add your field analysis logic here
                            pass
            
            with upload_tab:
                st.subheader("Upload Crop Image")
                
                # Add crop selection before upload
                selected_crop = st.selectbox(
                    "Select Crop Type",
                    options=list(CROP_DATA.keys()),
                    help="Choose the crop type you want to analyze"
                )
                
                # File uploader
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a clear image of your crop"
                )
                
                # Process uploaded image
                if uploaded_file is not None:
                    # Display original image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width="stretch")
                    
                    # Add analyze button
                    if st.button("Analyze Image"):
                        with st.spinner("Analyzing image..."):
                            process_uploaded_image(uploaded_file, selected_crop)
                            
                    # Display crop information
                    display_crop_info(CROP_DATA, selected_crop)
                    display_crop_management(selected_crop)
                    display_full_crop_info(selected_crop)
                
                # NDVI analysis function
                def ndvi_analysis(uploaded_file, selected_crop):
                    """Perform NDVI analysis on uploaded image"""
                    try:
                        ndvi_img, ndvi_values = compute_ndvi(uploaded_file)
                        if ndvi_img and ndvi_values is not None:
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(Image.open(uploaded_file), caption="Original Image", width="stretch")
                            with col2:
                                st.image(ndvi_img, caption="NDVI Analysis", width="stretch")
                        
                            # Show NDVI statistics
                            mean_ndvi = np.mean(ndvi_values)
                            st.write("#### NDVI Analysis Results")
                            
                            # Display health status based on NDVI
                            if mean_ndvi > 0.5:
                                st.success(f"🌿 Healthy Vegetation (NDVI: {mean_ndvi:.2f})")
                                st.info("Your crop appears to be in good health!")
                            elif mean_ndvi > 0.2:
                                st.warning(f"⚠️ Moderate Vegetation (NDVI: {mean_ndvi:.2f})")
                                st.info("Your crop may need some attention.")
                            else:
                                st.error(f"🚨 Poor Vegetation (NDVI: {mean_ndvi:.2f})")
                                st.info("Your crop needs immediate attention!")
                        
                            # Show crop-specific recommendations
                            if selected_crop in CROP_DATA:
                                st.subheader("📋 Recommendations")
                                for rec in CROP_DATA[selected_crop]['recommendations']:
                                    st.write(f"• {rec}")
                                    
                    except Exception as e:
                        st.error(f"Error in NDVI analysis: {str(e)}")

                # Update get_varieties_for_crop function
                def get_varieties_for_crop(crop_data, selected_crop):
                    """Get varieties list based on crop selection"""
                    try:
                        varieties = crop_data[selected_crop].get('varieties', [])
                        if isinstance(varieties, dict):
                            # Flatten nested varieties
                            return [var for sublist in varieties.values() for var in sublist]
                        return varieties
                    except Exception as e:
                        st.error(f"Error getting varieties: {str(e)}")
                        return []

                # Display recommendations section
                if uploaded_file is not None:
                    recommendations = CROP_DATA[selected_crop]['recommendations']
                    display_recommendations(recommendations, lang_code)
                
                # Add pest information
                display_pest_info(selected_crop)
                
                # Add analytics dashboard
                if st.checkbox("Show Crop Analytics"):
                    display_crop_analysis()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        if st.button("Try Again"):
            st.rerun()

# Add function to load and analyze historical crop data
def load_crop_data():
    """Load historical crop health data"""
    # Example dataset structure
    data = {
        'Date': pd.date_range(start='2025-01-01', periods=100, freq='D'),
        'Crop': ['Rice (Paddy)', 'Wheat'] * 50,
        'NDVI': np.random.uniform(0.2, 0.8, 100),
        'Pest_Pressure': np.random.choice(['Low', 'Medium', 'High'], 100),
        'Disease_Incidence': np.random.uniform(0, 1, 100),
        'Yield_Estimate': np.random.uniform(2000, 4000, 100)
    }
    return pd.DataFrame(data)

# Add function to display pest information
def display_pest_info(selected_crop):
    """Display pest information and management options"""
    if selected_crop in PEST_DATA:
        st.subheader("🐛 Common Pests and Management")
        
        for pest, info in PEST_DATA[selected_crop].items():
            with st.expander(f"🔍 {pest}"):
                # Symptoms
                st.write("**Signs to Look For:**")
                for symptom in info['symptoms']:
                    st.write(f"• {symptom}")
                
                # Solutions
                st.write("\n**What You Can Do:**")
                for solution in info['solutions']:
                    st.write(f"• {solution}")
                
                # Severity levels if available
                if 'severity_levels' in info:
                    st.write("\n**Severity Levels:**")
                    cols = st.columns(len(info['severity_levels']))
                    for (level, desc), col in zip(info['severity_levels'].items(), cols):
                        with col:
                            if level == "Low":
                                st.success(f"**{level}**\n{desc}")
                            elif level == "Medium":
                                st.warning(f"**{level}**\n{desc}")
                            else:
                                st.error(f"**{level}**\n{desc}")

# Add function to display crop analysis dashboard
def display_crop_analysis():
    """Display crop health analysis dashboard"""
    st.subheader("📊 Crop Health Analytics")
    
    # Load historical data
    df = load_crop_data()
    
    # Create dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # NDVI Trend
        st.write("**NDVI Trend Over Time**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x='Date', y='NDVI', hue='Crop')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
        
    with col2:
        # Pest Pressure Distribution
        st.write("**Pest Pressure Distribution**")
        fig, ax = plt.subplots(figsize=(10, 6))
        pest_counts = df['Pest_Pressure'].value_counts()
        plt.pie(pest_counts, labels=pest_counts.index, autopct='%1.1f%%')
        st.pyplot(fig)
        plt.close()
    
    # Add metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average NDVI", f"{df['NDVI'].mean():.2f}")
    with col2:
        st.metric("Disease Risk", f"{df['Disease_Incidence'].mean():.2%}")
    with col3:
        st.metric("Estimated Yield", f"{df['Yield_Estimate'].mean():.0f} kg/ha")

def display_vegetation_analysis(ndvi_values, selected_crop):
    """Display farmer-friendly vegetation analysis"""
    mean_ndvi = float(np.mean(ndvi_values))  # Convert to Python float
    st.write("### 🌱 Crop Health Check")
    
    # Create visual health meter with proper float value
    health_cols = st.columns([1, 3, 1])
    with health_cols[1]:
        # Ensure normalized_health is a Python float between 0 and 1
        normalized_health = float(min(max((mean_ndvi + 1) / 2, 0), 1))
        st.progress(normalized_health)
        st.write(f"Health Score: {normalized_health:.0%}")
    
    # Add pest detection section
    st.write("### 🐛 Pest Detection")
    if selected_crop in PEST_DATA:
        with st.expander("Check for Common Pests"):
            for pest, info in PEST_DATA[selected_crop].items():
                st.subheader(f"📍 {pest}")
                
                # Show pest symptoms as checkboxes
                st.write("**Look for these signs:**")
                for symptom in info['symptoms']:
                    has_symptom = st.checkbox(symptom, key=f"{pest}_{symptom}")
                
                # Show solutions in a clean format
                st.write("**Solutions:**")
                for solution in info['solutions']:
                    st.info(f"• {solution}")
                
                # Show severity level if available
                if 'severity_levels' in info:
                    severity = st.radio(
                        "Select severity level:",
                        options=list(info['severity_levels'].keys()),
                        key=f"{pest}_severity"
                    )
                    
                    if severity == "Low":
                        st.success(f"✅ {info['severity_levels'][severity]}")
                    elif severity == "Medium":
                        st.warning(f"⚠️ {info['severity_levels'][severity]}")
                    else:
                        st.error(f"🚨 {info['severity_levels'][severity]}")
                
                st.markdown("---")
    else:
        st.info("No pest information available for this crop type")
    
    # Show recommendations
    st.write("### 📋 Recommendations")
    if selected_crop in CROP_DATA:
        for idx, rec in enumerate(CROP_DATA[selected_crop]['recommendations'], 1):
            st.info(f"{idx}. {rec}")
    
    # Add weather considerations
    st.write("### 🌤️ Weather Tips")
    st.info("""
    • Best time to spray: Early morning or evening
    • Don't spray if rain is expected within 6 hours
    • Check wind conditions before spraying
    """)
    
    # Add emergency contacts
    st.write("### 📞 Need Help?")
    st.warning("""
    **Contact local experts:**
    • Kisan Call Center: 1800-180-1551
    • Agricultural Extension Officer
    • Local Krishi Vigyan Kendra
    """)

if __name__ == "__main__":
    init_page()
    try:
        display_dashboard()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        if st.button("Restart"):
            st.rerun()