# 7_advanced_multi_digit_recognizer.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import imutils
import os
from typing import List, Tuple
import pytesseract
from pytesseract import Output
import easyocr
import io

# Page config
st.set_page_config(
    page_title="Advanced Multi-Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)

# Title
st.title("🔢 Advanced Multi-Digit Recognizer")
st.markdown("### Recognizes multiple digits in a single image!")

# Load model
@st.cache_resource
def load_my_model():
    try:
        possible_models = [
            'best_cnn_model.h5',
            'mlp_digit_recognizer.h5',
            'cnn_digit_recognizer.h5'
        ]
        
        for model_path in possible_models:
            if os.path.exists(model_path):
                model = load_model(model_path)
                return model
        return None
    except:
        return None

model = load_my_model()

if model is None:
    st.error("❌ No model found! Please train a model first.")
    st.stop()

# Initialize EasyOCR (optional, better than Tesseract)
@st.cache_resource
def load_easyocr():
    try:
        reader = easyocr.Reader(['en'])
        return reader
    except:
        return None

reader = load_easyocr()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    detection_method = st.selectbox(
        "Detection Method",
        ["Contour Detection", "EasyOCR (Deep Learning)", "Tesseract OCR", "Sliding Window"]
    )
    
    st.markdown("---")
    st.header("🎛️ Parameters")
    
    if detection_method == "Contour Detection":
        threshold_value = st.slider("Threshold Value", 0, 255, 150)
        min_area = st.slider("Minimum Contour Area", 50, 500, 100)
        padding = st.slider("Padding around digits", 2, 20, 5)
    
    elif detection_method == "Sliding Window":
        window_size = st.slider("Window Size", 20, 50, 28)
        stride = st.slider("Stride", 5, 20, 10)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    This advanced version can detect and recognize **multiple digits** in a single image!
    
    **Features:**
    - Multiple detection methods
    - Bounding box visualization
    - Sorts digits left-to-right
    - Confidence scores for each
    - Works on complex images
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input Image")
    uploaded_file = st.file_uploader(
        "Upload an image with digits",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Read and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Process button
        if st.button("🔍 Detect & Recognize Digits", type="primary", use_container_width=True):
            with st.spinner("Processing image..."):
                
                # Store results
                detected_digits = []
                bounding_boxes = []
                
                if detection_method == "Contour Detection":
                    # Thresholding
                    thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
                    
                    # Find contours
                    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = imutils.grab_contours(contours)
                    
                    # Filter and sort contours (left to right)
                    digit_contours = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > min_area:
                            x, y, w, h = cv2.boundingRect(cnt)
                            aspect_ratio = h / w if w > 0 else 0
                            if 0.5 < aspect_ratio < 3.0:  # Filter by aspect ratio
                                digit_contours.append((x, cnt))
                    
                    digit_contours.sort(key=lambda x: x[0])  # Sort left to right
                    
                    # Process each digit
                    result_img = img_cv.copy()
                    for i, (x, cnt) in enumerate(digit_contours):
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # Add padding
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(gray.shape[1] - x, w + 2*padding)
                        h = min(gray.shape[0] - y, h + 2*padding)
                        
                        # Extract digit
                        digit_roi = gray[y:y+h, x:x+w]
                        
                        # Preprocess for model
                        digit_resized = cv2.resize(digit_roi, (28, 28))
                        digit_resized = cv2.bitwise_not(digit_resized)  # Invert
                        digit_array = digit_resized.astype('float32') / 255.0
                        digit_array = digit_array.reshape(1, 28, 28, 1)
                        
                        # Predict
                        pred = model.predict(digit_array, verbose=0)[0]
                        digit = np.argmax(pred)
                        confidence = pred[digit]
                        
                        detected_digits.append((digit, confidence))
                        bounding_boxes.append((x, y, w, h))
                        
                        # Draw bounding box and prediction
                        color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                        label = f"{digit} ({confidence:.2f})"
                        cv2.putText(result_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    st.session_state['result_img'] = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.session_state['detected_digits'] = detected_digits
                    st.session_state['bounding_boxes'] = bounding_boxes
                
                elif detection_method == "EasyOCR (Deep Learning)" and reader:
                    # Use EasyOCR
                    results = reader.readtext(img_cv)
                    
                    result_img = img_cv.copy()
                    for (bbox, text, confidence) in results:
                        if text.isdigit() and len(text) > 0:
                            for i, digit_char in enumerate(text):
                                digit = int(digit_char)
                                detected_digits.append((digit, confidence))
                                
                                # Approximate bounding box for each digit
                                if len(text) > 1:
                                    x_vals = [point[0] for point in bbox]
                                    y_vals = [point[1] for point in bbox]
                                    x_min, x_max = min(x_vals), max(x_vals)
                                    y_min, y_max = min(y_vals), max(y_vals)
                                    
                                    digit_width = (x_max - x_min) / len(text)
                                    x1 = int(x_min + i * digit_width)
                                    x2 = int(x_min + (i + 1) * digit_width)
                                    
                                    cv2.rectangle(result_img, (x1, int(y_min)), (x2, int(y_max)), (0, 255, 0), 2)
                                    cv2.putText(result_img, digit_char, (x1, int(y_min)-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                else:
                                    points = np.array(bbox, dtype=np.int32)
                                    cv2.polylines(result_img, [points], True, (0, 255, 0), 2)
                                    cv2.putText(result_img, digit_char, tuple(bbox[0].astype(int)), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    st.session_state['result_img'] = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.session_state['detected_digits'] = detected_digits
                
                elif detection_method == "Sliding Window":
                    # Sliding window approach
                    h, w = gray.shape
                    result_img = img_cv.copy()
                    
                    for y in range(0, h - window_size, stride):
                        for x in range(0, w - window_size, stride):
                            window = gray[y:y+window_size, x:x+window_size]
                            
                            # Resize to 28x28
                            window_resized = cv2.resize(window, (28, 28))
                            window_array = window_resized.astype('float32') / 255.0
                            window_array = 1.0 - window_array  # Invert
                            window_array = window_array.reshape(1, 28, 28, 1)
                            
                            # Predict
                            pred = model.predict(window_array, verbose=0)[0]
                            confidence = np.max(pred)
                            
                            if confidence > confidence_threshold:
                                digit = np.argmax(pred)
                                detected_digits.append((digit, confidence))
                                bounding_boxes.append((x, y, window_size, window_size))
                                
                                cv2.rectangle(result_img, (x, y), (x+window_size, y+window_size), (0, 255, 0), 1)
                                cv2.putText(result_img, str(digit), (x, y-2), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Non-maximum suppression to remove overlapping boxes
                    if len(bounding_boxes) > 0:
                        indices = cv2.dnn.NMSBoxes(bounding_boxes, [c for _, c in detected_digits], 0.5, 0.4)
                        if len(indices) > 0:
                            indices = indices.flatten()
                            detected_digits = [detected_digits[i] for i in indices]
                    
                    st.session_state['result_img'] = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.session_state['detected_digits'] = detected_digits

with col2:
    st.subheader("📊 Results")
    
    if 'result_img' in st.session_state:
        # Display result image
        st.image(st.session_state['result_img'], caption="Detected Digits", use_column_width=True)
        
        # Display detected digits
        if 'detected_digits' in st.session_state and len(st.session_state['detected_digits']) > 0:
            st.success(f"✅ Found {len(st.session_state['detected_digits'])} digits!")
            
            # Format as number
            digits_str = ''.join([str(d) for d, _ in st.session_state['detected_digits']])
            st.markdown(f"### 🔢 Detected Number: **{digits_str}**")
            
            # Create dataframe for display
            import pandas as pd
            df_data = []
            for i, (digit, conf) in enumerate(st.session_state['detected_digits']):
                df_data.append({
                    "Position": i+1,
                    "Digit": digit,
                    "Confidence": f"{conf*100:.2f}%",
                    "Status": "✅ High" if conf > 0.8 else "⚠️ Medium" if conf > 0.5 else "❌ Low"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Confidence chart
            fig, ax = plt.subplots(figsize=(10, 4))
            digits = [d for d, _ in st.session_state['detected_digits']]
            confs = [c*100 for _, c in st.session_state['detected_digits']]
            positions = range(1, len(digits)+1)
            
            colors = ['green' if c > 80 else 'orange' if c > 50 else 'red' for c in confs]
            bars = ax.bar(positions, confs, color=colors)
            ax.set_xlabel('Digit Position')
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Confidence per Digit')
            ax.set_xticks(positions)
            ax.set_xticklabels([f"Pos {i}" for i in positions])
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bar, conf, digit in zip(bars, confs, digits):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{digit}\n({conf:.1f}%)', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
            
            # Export options
            st.markdown("### 📥 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button(
                    "📄 Download as Text",
                    data=digits_str,
                    file_name="detected_number.txt",
                    mime="text/plain"
                )
            with col_exp2:
                # Save result image
                result_pil = Image.fromarray(st.session_state['result_img'])
                buf = io.BytesIO()
                result_pil.save(buf, format='PNG')
                st.download_button(
                    "🖼️ Download Image",
                    data=buf.getvalue(),
                    file_name="detected_digits.png",
                    mime="image/png"
                )
        else:
            st.warning("No digits detected in the image!")
    else:
        st.info("Upload an image and click 'Detect & Recognize Digits' to see results!")

# Advanced Features Section
st.markdown("---")
st.header("🚀 Advanced Features")

tab1, tab2, tab3 = st.tabs(["Batch Processing", "Real-time Camera", "API Mode"])

with tab1:
    st.subheader("📁 Batch Processing")
    batch_files = st.file_uploader("Upload multiple images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if batch_files and st.button("Process All Images"):
        with st.spinner("Processing multiple images..."):
            for file in batch_files:
                st.write(f"Processing: {file.name}")
                # Process each image (simplified for demo)

with tab2:
    st.subheader("📸 Real-time Camera")
    st.warning("Camera feature requires additional setup")
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        st.image(camera_input)

with tab3:
    st.subheader("🌐 API Mode")
    st.code("""
    # Python API client example
    import requests
    import base64
    
    with open('image.jpg', 'rb') as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()
    
    response = requests.post('http://localhost:8501/api/predict', 
                            json={'image': img_base64})
    print(response.json()['digits'])
    """, language='python')

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Advanced Multi-Digit Recognizer | Built with Streamlit, TensorFlow, and OpenCV</div>",
    unsafe_allow_html=True
)