import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import io
import joblib
from sklearn.preprocessing import StandardScaler
import time
from streamlit_drawable_canvas import st_canvas
import base64
from datetime import datetime
import json
import cv2
from scipy import ndimage
import random

# Set page config with a modern title and icon
st.set_page_config(
    page_title="AI Digit Recognition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern dark theme with glassmorphism
st.markdown("""
    <style>
    /* Modern theme colors with better contrast */
    :root {
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --accent-primary: #6366F1;
        --accent-secondary: #818CF8;
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --success: #22C55E;
        --warning: #F59E0B;
        --error: #EF4444;
        --border: rgba(255, 255, 255, 0.15);
        --glass: rgba(255, 255, 255, 0.05);
    }

    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0F172A, #1E293B) !important;
        color: var(--text-primary) !important;
    }

    /* Main container */
    .block-container {
        padding: 3rem 5% !important;
        max-width: 1400px !important;
    }

    /* Glassmorphism cards with better visibility */
    .stTabs, .prediction-box, .canvas-container, .stFileUploader, div[data-testid="stExpander"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 1rem !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Headers with brighter gradients */
    .app-title {
        background: linear-gradient(135deg, #818CF8, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        padding: 1rem;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }

    .app-subtitle {
        color: var(--text-secondary) !important;
        text-align: center;
        font-size: 1.2rem !important;
        margin-bottom: 2rem !important;
        font-weight: 400 !important;
    }

    /* Drawing section with better contrast */
    .drawing-section h2 {
        background: linear-gradient(135deg, #818CF8, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }

    .drawing-instructions {
        color: var(--text-secondary) !important;
        font-size: 1.1rem !important;
        margin-bottom: 2rem !important;
    }

    /* Canvas with better visibility */
    [data-testid="stCanvas"] {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border) !important;
        border-radius: 1rem !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        margin: 0 auto !important;
        display: flex !important;
        justify-content: center !important;
        max-width: 500px !important;
    }

    canvas {
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        border: 2px solid var(--border) !important;
    }

    /* Buttons with brighter colors */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #818CF8) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(135deg, #818CF8, #6366F1) !important;
    }

    /* Prediction display with better visibility */
    .prediction-box {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        text-align: center !important;
        margin: 2rem auto !important;
        max-width: 600px !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .prediction-number {
        font-size: 10rem !important;
        font-weight: 900 !important;
        line-height: 1 !important;
        margin: 1rem 0 !important;
        background: linear-gradient(135deg, #818CF8, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }

    .confidence-container {
        background: var(--bg-primary) !important;
        border-radius: 1rem !important;
        padding: 1rem !important;
        margin: 1rem auto !important;
        max-width: 400px !important;
        border: 1px solid var(--border) !important;
    }

    .confidence-bar {
        height: 10px !important;
        background: linear-gradient(90deg, #6366F1, #818CF8) !important;
        border-radius: 5px !important;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;
    }

    .confidence-text {
        font-size: 1.25rem !important;
        color: var(--text-secondary) !important;
        margin-top: 0.5rem !important;
        font-weight: 500 !important;
    }

    /* Status messages with better contrast */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        padding: 1rem !important;
        border-radius: 0.75rem !important;
        margin: 1rem 0 !important;
        font-weight: 500 !important;
        text-align: center !important;
    }

    .stSuccess {
        border-left: 4px solid var(--success) !important;
    }

    .stInfo {
        border-left: 4px solid var(--accent-primary) !important;
    }

    .stWarning {
        border-left: 4px solid var(--warning) !important;
    }

    .stError {
        border-left: 4px solid var(--error) !important;
    }

    /* Tabs with better visibility */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-primary) !important;
        padding: 0.5rem !important;
        border-radius: 0.75rem !important;
        gap: 0.5rem !important;
        border: 1px solid var(--border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #818CF8) !important;
        color: white !important;
        border: none !important;
    }

    /* File uploader with better contrast */
    .stFileUploader {
        background: var(--bg-secondary) !important;
        border: 2px dashed var(--border) !important;
    }

    .stFileUploader [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        # Load ensemble model
        model = joblib.load('models/digit_classifier_ensemble.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # Verify model is loaded correctly
        if model is None or not hasattr(model, 'predict'):
            raise ValueError("Invalid model loaded")
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_canvas_data(canvas_result):
    """Preprocess the canvas data for prediction"""
    # Get image data from canvas
    img = canvas_result.image_data
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # Apply thresholding to segment the digit
    _, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour (the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Extract the digit and add padding
    padding = 20
    digit = img_thresh[max(0, y-padding):min(img_thresh.shape[0], y+h+padding),
                      max(0, x-padding):min(img_thresh.shape[1], x+w+padding)]
    
    # Resize to 28x28 (MNIST format)
    digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    digit_normalized = digit_resized / 255.0
    
    # Flatten the image
    digit_flattened = digit_normalized.reshape(1, -1)
    
    return digit_flattened

def preprocess_image(image, preprocessing_options):
    """Preprocess the uploaded image with additional options"""
    try:
        # Convert to grayscale
        img_gray = image.convert('L')
        
        # Apply preprocessing options
        if preprocessing_options.get('enhance_contrast', False):
            img_gray = ImageEnhance.Contrast(img_gray).enhance(2)
        
        if preprocessing_options.get('sharpen', False):
            img_gray = img_gray.filter(ImageFilter.SHARPEN)
        
        if preprocessing_options.get('denoise', False):
            img_gray = img_gray.filter(ImageFilter.MedianFilter(size=3))
        
        # Convert to numpy array
        img_array = np.array(img_gray)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        # Invert if specified
        if preprocessing_options.get('invert', False):
            img_array = 1 - img_array
        
        # Reshape for model input
        img_array = img_array.reshape(1, -1)
        
        return img_array, img_gray
    except Exception as e:
        st.error(f"Error in preprocessing image: {str(e)}")
        return np.zeros((1, 784)), None

def export_session_data():
    """Export session data as JSON"""
    if st.session_state.prediction_history:
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': st.session_state.prediction_history,
            'summary': {
                'total_predictions': len(st.session_state.prediction_history),
                'average_confidence': sum(p['confidence'] for p in st.session_state.prediction_history) / len(st.session_state.prediction_history),
                'digit_distribution': {str(d): st.session_state.prediction_history.count(d) for d in range(10)}
            }
        }
        return json.dumps(data, indent=2)
    return None

def show_image_preprocessing_options():
    """Show and return image preprocessing options"""
    st.write("Image Preprocessing Options:")
    col1, col2 = st.columns(2)
    
    with col1:
        enhance_contrast = st.checkbox("Enhance Contrast", value=False)
        sharpen = st.checkbox("Sharpen", value=False)
    
    with col2:
        denoise = st.checkbox("Denoise", value=False)
        invert = st.checkbox("Invert Colors", value=False)
    
    return {
        'enhance_contrast': enhance_contrast,
        'sharpen': sharpen,
        'denoise': denoise,
        'invert': invert
    }

def show_model_explanation(digit, image):
    """Show model explanation for the prediction"""
    st.subheader("Model Explanation")
    
    # Show preprocessed image
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Preprocessed Image")
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        st.pyplot(plt)
    
    with col2:
        st.write("Pixel Intensity Heatmap")
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='hot')
        plt.colorbar()
        plt.axis('off')
        st.pyplot(plt)
    
    # Show feature importance
    st.write("Feature Importance")
    feature_imp = np.abs(image - np.mean(image))
    plt.figure(figsize=(10, 4))
    plt.plot(feature_imp.flatten())
    plt.title("Pixel Importance")
    plt.xlabel("Pixel Position")
    plt.ylabel("Importance")
    st.pyplot(plt)

def make_prediction(model, scaler, image_array):
    """Make prediction using the model with confidence threshold"""
    try:
        # Ensure the input is the right shape
        if len(image_array.shape) == 2:
            image_array = image_array.reshape(1, -1)
        
        # Replace any NaN values
        image_array = np.nan_to_num(image_array, nan=0.0)
        
        # Scale the image data
        image_scaled = scaler.transform(image_array)
        
        # Get predictions from all models in ensemble
        probabilities = model.predict_proba(image_scaled)[0]
        
        # Get the predicted digit
        digit = np.argmax(probabilities)
        
        # If confidence is too low, return None
        if probabilities[digit] < 0.4:  # Minimum confidence threshold
            return None, probabilities
        
        return digit, probabilities
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, np.zeros(10)

def show_prediction(digit, probabilities):
    """Display prediction results with modern visualization"""
    confidence = probabilities[digit]
    
    st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-number">{digit}</div>
            <div class="confidence-container">
                <div class="confidence-bar" style="width: {confidence * 100}%"></div>
                <div class="confidence-text">Confidence: {confidence:.1%}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if confidence > 0.8:
        st.success("")
    elif confidence > 0.5:
        st.info("")
    else:
        st.warning("")

def main():
    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # Load model
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Could not load model")
        return

    # App title with modern styling
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">✨ AI Digit Recognition</h1>
            <p class="app-subtitle">Draw or upload digits and watch the AI predict them!</p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs with modern styling
    tab1, tab2, tab3 = st.tabs(["✏️ Draw", "📤 Upload", "📊 History"])

    with tab1:
        st.markdown("""
            <div class="drawing-section">
                <div class="instruction-box">
                    <div class="instruction-header">
                        <h2>✏️ Draw a Digit</h2>
                    </div>
                    <div class="instruction-steps">
                        <p>1. Use the canvas below to draw</p>
                        <p>2. Draw a single digit (0-9)</p>
                        <p>3. Make it large and clear</p>
                        <p>4. Click 'Predict' when ready</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Create canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=25,
            stroke_color="#FFFFFF",
            background_color="#1E293B",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
        )

        # Add drawing controls with better layout
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Canvas", use_container_width=True):
                st.session_state['canvas_key'] = str(random.randint(0, 1000000))
                st.experimental_rerun()
        with col2:
            if st.button("✨ Predict", use_container_width=True):
                if canvas_result.image_data is not None:
                    img_array = preprocess_canvas_data(canvas_result)
                    if img_array is not None:
                        digit, probabilities = make_prediction(model, scaler, img_array)
                        if digit is not None:
                            show_prediction(digit, probabilities)
                            
                            if st.button("💾 Save to History"):
                                st.session_state.prediction_history.append({
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "predicted_digit": int(digit),
                                    "confidence": probabilities[digit] * 100
                                })
                                st.rerun()
                        else:
                            st.error("Could not make a prediction. Please try drawing again.")
                    else:
                        st.warning("Could not detect a digit. Please draw more clearly.")
                else:
                    st.info("Please draw a digit first!")

    # Upload tab
    with tab2:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            try:
                # Process uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Convert to numpy array
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Resize and process
                img_array = cv2.resize(img_array, (28, 28))
                img_array = img_array.astype('float32') / 255.0
                
                # Make prediction
                digit, probabilities = make_prediction(model, scaler, img_array.reshape(1, -1))
                
                if digit is not None:
                    show_prediction(digit, probabilities)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    # History tab
    with tab3:
        st.subheader("Prediction History")
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions in history yet. Draw or upload some digits!")

    st.markdown("""
        <style>
        /* App header */
        .app-header {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        /* Drawing section */
        .instruction-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        .instruction-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .instruction-header h2 {
            background: linear-gradient(135deg, #818CF8, #6366F1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }

        .instruction-steps {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.5rem;
        }

        .instruction-steps p {
            color: var(--text-primary) !important;
            font-size: 1.1rem !important;
            margin: 0.75rem 0 !important;
            padding-left: 1.5rem !important;
            position: relative !important;
        }

        .instruction-steps p::before {
            content: '•';
            position: absolute;
            left: 0.5rem;
            color: var(--accent-primary);
        }

        /* Canvas container */
        [data-testid="stCanvas"] {
            background: var(--bg-secondary) !important;
            border: 2px solid var(--border) !important;
            border-radius: 1rem !important;
            padding: 2rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
            margin: 0 auto !important;
            display: flex !important;
            justify-content: center !important;
            max-width: 500px !important;
        }

        /* Canvas */
        canvas {
            border-radius: 0.75rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
            border: 2px solid rgba(255, 255, 255, 0.1) !important;
            transition: all 0.3s ease !important;
        }

        canvas:hover {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.3) !important;
        }

        /* Control buttons */
        .stButton > button {
            background: linear-gradient(135deg, #6366F1, #818CF8) !important;
            color: white !important;
            border: none !important;
            padding: 1rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            border-radius: 0.75rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
            margin: 1rem 0 !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
            background: linear-gradient(135deg, #818CF8, #6366F1) !important;
        }

        /* Status messages */
        .stSuccess, .stInfo, .stWarning, .stError {
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            padding: 1rem 1.5rem !important;
            border-radius: 0.75rem !important;
            margin: 1rem 0 !important;
            font-weight: 500 !important;
            text-align: center !important;
            font-size: 1.1rem !important;
        }

        .stSuccess {
            border-left: 4px solid var(--success) !important;
        }

        .stInfo {
            border-left: 4px solid var(--accent-primary) !important;
        }

        .stWarning {
            border-left: 4px solid var(--warning) !important;
        }

        .stError {
            border-left: 4px solid var(--error) !important;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
