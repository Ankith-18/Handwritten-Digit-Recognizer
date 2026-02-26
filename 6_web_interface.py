# 6_web_interface.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io

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
        model = load_model('cnn_digit_recognizer.h5')
        return model
    except:
        return None

model = load_my_model()

if model is None:
    st.error("Model not found! Please train the model first.")
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
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=280)
            
            # Preprocess button
            if st.button("Predict", type="primary"):
                with st.spinner("Analyzing..."):
                    # Preprocess image
                    img = image.convert('L')
                    img = img.resize((28, 28))
                    img_array = np.array(img)
                    img_array = 255 - img_array  # Invert
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
    
    else:  # Draw Digit
        st.markdown("Draw a digit in the canvas below:")
        
        # Canvas for drawing
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
            if st.button("Predict", type="primary"):
                with st.spinner("Analyzing..."):
                    # Convert canvas to image
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                    img = img.convert('L')
                    img = img.resize((28, 28))
                    img_array = np.array(img)
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

with col2:
    st.subheader("Results")
    
    if 'predicted_digit' in st.session_state:
        # Display prediction
        st.markdown(
            f"<h2 style='text-align: center;'>Predicted Digit: {st.session_state['predicted_digit']}</h2>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"<h3 style='text-align: center; color: green;'>Confidence: {st.session_state['confidence']:.2f}%</h3>",
            unsafe_allow_html=True
        )
        
        # Display probability chart
        st.subheader("Probability Distribution")
        
        if 'predictions' in st.session_state:
            fig, ax = plt.subplots(figsize=(10, 4))
            digits = range(10)
            probs = st.session_state['predictions'] * 100
            
            colors = ['blue'] * 10
            colors[st.session_state['predicted_digit']] = 'green'
            
            bars = ax.bar(digits, probs, color=colors)
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability (%)')
            ax.set_title('Prediction Probabilities')
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
    else:
        st.info("Upload an image or draw a digit to see predictions!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow | MNIST Dataset")