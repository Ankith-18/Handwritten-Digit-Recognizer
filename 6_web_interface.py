# 6_web_interface.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import os  # This was missing!

# Page config
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)

# Title
st.title("🔢 Handwritten Digit Recognizer")
st.markdown("---")

# Load model
@st.cache_resource
def load_my_model():
    try:
        # Try multiple possible model filenames in order of preference
        possible_models = [
            'best_cnn_model.h5',      # Best model (99.48% accuracy)
            'mlp_digit_recognizer.h5', # MLP model (98% accuracy)
            'cnn_digit_recognizer.h5'  # Alternative CNN name
        ]
        
        for model_path in possible_models:
            if os.path.exists(model_path):
                model = load_model(model_path)
                st.sidebar.success(f"✅ Loaded: {model_path}")
                st.sidebar.info(f"Model size: {os.path.getsize(model_path)/(1024*1024):.1f} MB")
                return model
        
        st.sidebar.error("❌ No model found! Please train a model first.")
        st.sidebar.info("Run: python 4_cnn_model.py")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

model = load_my_model()

if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app recognizes handwritten digits (0-9) using a 
    Convolutional Neural Network trained on the MNIST dataset.
    
    **Model Architecture:**
    - Multiple Conv2D layers
    - Batch Normalization
    - MaxPooling
    - Dropout for regularization
    - Dense layers for classification
    
    **Accuracy:** ~99% on test set
    """)
    
    st.markdown("---")
    st.header("Instructions")
    st.markdown("""
    1. Upload an image of a handwritten digit
    2. Or draw a digit using the canvas
    3. Click 'Predict' to see the result
    """)
    
    st.markdown("---")
    st.header("Model Info")
    if model is not None:
        st.success("✅ Model is loaded and ready!")
        
        # Try to show model summary (optional)
        try:
            total_params = model.count_params()
            st.info(f"Total parameters: {total_params:,}")
        except:
            pass

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Draw Digit"]
    )
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a digit image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=280)
            
            # Preprocess button
            if st.button("Predict", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Preprocess image
                    img = image.convert('L')  # Convert to grayscale
                    img = img.resize((28, 28))
                    img_array = np.array(img)
                    img_array = 255 - img_array  # Invert (MNIST has white digits on black background)
                    img_array = img_array.astype('float32') / 255.0
                    img_array = img_array.reshape(1, 28, 28, 1)
                    
                    # Predict
                    predictions = model.predict(img_array, verbose=0)[0]
                    predicted_digit = np.argmax(predictions)
                    confidence = predictions[predicted_digit] * 100
                    
                    # Store in session state
                    st.session_state['predictions'] = predictions
                    st.session_state['predicted_digit'] = predicted_digit
                    st.session_state['confidence'] = confidence
                    
                    # Also store the image for display
                    st.session_state['input_image'] = image
    
    else:  # Draw Digit
        st.markdown("Draw a digit in the canvas below:")
        
        # Check if streamlit-drawable-canvas is installed
        try:
            from streamlit_drawable_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=15,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas"
            )
            
            if canvas_result.image_data is not None:
                if st.button("Predict", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        # Convert canvas to image
                        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                        img = img.convert('L')
                        img = img.resize((28, 28))
                        img_array = np.array(img)
                        img_array = img_array.astype('float32') / 255.0
                        # Invert if needed (depends on canvas background)
                        if img_array.mean() > 0.5:  # If background is light
                            img_array = 1.0 - img_array
                        img_array = img_array.reshape(1, 28, 28, 1)
                        
                        # Predict
                        predictions = model.predict(img_array, verbose=0)[0]
                        predicted_digit = np.argmax(predictions)
                        confidence = predictions[predicted_digit] * 100
                        
                        # Store in session state
                        st.session_state['predictions'] = predictions
                        st.session_state['predicted_digit'] = predicted_digit
                        st.session_state['confidence'] = confidence
                        
        except ImportError:
            st.error("⚠️ Drawing canvas requires: pip install streamlit-drawable-canvas")
            st.info("Please run: pip install streamlit-drawable-canvas")
            
            # Fallback to file upload
            st.warning("Using file upload as fallback. Install the package for drawing.")
            uploaded_file = st.file_uploader(
                "Upload a digit image instead",
                type=['png', 'jpg', 'jpeg', 'bmp']
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, width=280)
                if st.button("Predict Uploaded"):
                    # Same prediction code as above
                    img = image.convert('L').resize((28, 28))
                    img_array = 255 - np.array(img)
                    img_array = img_array.astype('float32') / 255.0
                    img_array = img_array.reshape(1, 28, 28, 1)
                    
                    predictions = model.predict(img_array, verbose=0)[0]
                    st.session_state['predictions'] = predictions
                    st.session_state['predicted_digit'] = np.argmax(predictions)
                    st.session_state['confidence'] = predictions[np.argmax(predictions)] * 100

with col2:
    st.subheader("Results")
    
    if 'predicted_digit' in st.session_state:
        # Create a nice container for results
        with st.container():
            # Display prediction in a big box
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: #1f77b4; font-size: 48px;">{st.session_state['predicted_digit']}</h1>
                    <h3 style="color: green;">Confidence: {st.session_state['confidence']:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display probability chart
        st.subheader("📊 Probability Distribution")
        
        if 'predictions' in st.session_state:
            fig, ax = plt.subplots(figsize=(10, 5))
            digits = range(10)
            probs = st.session_state['predictions'] * 100
            
            colors = ['#1f77b4'] * 10
            colors[st.session_state['predicted_digit']] = '#2ecc71'  # Green for predicted
            
            bars = ax.bar(digits, probs, color=colors, edgecolor='black', linewidth=1)
            ax.set_xlabel('Digit', fontsize=12)
            ax.set_ylabel('Probability (%)', fontsize=12)
            ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
            
            # Show top 3 predictions
            top_3_idx = np.argsort(st.session_state['predictions'])[-3:][::-1]
            top_3_probs = st.session_state['predictions'][top_3_idx] * 100
            
            st.subheader("🔝 Top 3 Predictions")
            cols = st.columns(3)
            for i, (idx, prob) in enumerate(zip(top_3_idx, top_3_probs)):
                with cols[i]:
                    st.metric(f"Digit {idx}", f"{prob:.1f}%")
    else:
        st.info("👆 Upload an image or draw a digit to see predictions!")
        
        # Show example
        st.subheader("📝 Example")
        st.markdown("""
        The model expects:
        - Single digit (0-9)
        - Centered in the image
        - Clear handwriting
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit and TensorFlow | MNIST Dataset</div>",
    unsafe_allow_html=True
)

# Add a reset button in sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Reset", use_container_width=True):
        if 'predicted_digit' in st.session_state:
            del st.session_state['predicted_digit']
        if 'predictions' in st.session_state:
            del st.session_state['predictions']
        if 'confidence' in st.session_state:
            del st.session_state['confidence']
        if 'input_image' in st.session_state:
            del st.session_state['input_image']
        st.rerun()