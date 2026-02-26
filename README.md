📝 Project Overview
This project implements a Handwritten Digit Recognition System using Deep Learning. It can recognize handwritten digits (0-9) from images or real-time drawing with 99.48% accuracy. The model is trained on the famous MNIST dataset and deployed with both desktop GUI and web interfaces.

🎯 Main Objective
The primary goal of this project is to build a production-ready digit recognition system that demonstrates:
Deep Learning fundamentals - Building and training neural networks
Computer Vision - Image preprocessing and augmentation
Model Deployment - Creating user-friendly interfaces
End-to-end ML Pipeline - From data to deployment

<img width="610" height="426" alt="image" src="https://github.com/user-attachments/assets/3d462324-ed87-4b89-9eb5-b88f13f3afb6" />

🧠 Models Implemented
1. Simple MLP (Multilayer Perceptron)
Architecture: 3 Dense layers (512 → 256 → 128) + Dropout
Accuracy: ~98%
Purpose: Baseline model for comparison

2. Advanced CNN (Convolutional Neural Network) - BEST MODEL
Architecture:

3 Convolutional blocks (32→64→128 filters)
Batch Normalization after each conv layer
MaxPooling for dimensionality reduction
Dropout for regularization (0.25-0.5)
Dense layers (256→128) for classification
Accuracy: 99.48% on test set
Parameters: 619,114 trainable parameters


💻 Technologies Used
Programming Languages
Python 3.8+ - Core programming language

<img width="867" height="392" alt="image" src="https://github.com/user-attachments/assets/c99b7e13-46e9-42ce-b86b-95128a9f897b" />
<img width="819" height="342" alt="image" src="https://github.com/user-attachments/assets/ec5e78b9-6f81-433a-8eab-fa2443d7e44f" />

🔄 How It Works
Step 1: Data Loading
MNIST dataset (60,000 training + 10,000 test images)
Each image: 28×28 pixels, grayscale
Digits: 0-9 (10 classes)

Step 2: Preprocessing
# Normalization
X_train = X_train.astype('float32') / 255.0

# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train = to_categorical(y_train, 10)

Step 3: Data Augmentation
Rotation (±10°)
Zoom (±10%)
Width/height shifts (±10%)
Creates more training variations

Step 4: Model Training
CNN learns hierarchical features
Early stopping prevents overfitting
Learning rate reduction on plateau

Step 5: Prediction Pipeline
User Input (draw/upload) → Preprocess (28×28, normalize) 
→ Model Inference → Probability Distribution → Final Prediction

🚀 Features
✅ Real-time digit recognition
✅ Multiple input methods (draw, upload)
✅ Confidence scores for predictions
✅ Probability distribution visualization
✅ Top-3 predictions display
✅ Both GUI and Web interfaces
✅ Data augmentation for better accuracy
✅ Early stopping to prevent overfitting

<img width="634" height="236" alt="image" src="https://github.com/user-attachments/assets/d6ddc72e-d15b-4506-a655-e527a5a863f1" />

🛠️ Installation Guide for Contributors
If you fork this repository, follow these steps:

Prerequisites
Python 3.8 or higher
pip (Python package manager)
Git
4GB+ RAM recommended

Step-by-Step Setup
1. Clone the Repository
git clone https://github.com/Ankith-18/Handwritten-Digit-Recognizer.git
cd Handwritten-Digit-Recognizer

2. Create Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Additional Package for Drawing (Web App)
pip install streamlit-drawable-canvas

5. Run Data Preprocessing
python 2_data_preprocessing.py

6. Train Models (Optional - Models are included)
# Train CNN model (best - 10-15 min)
python 4_cnn_model.py

# OR train MLP model (faster - 2-3 min)
python 3_simple_model.py

7. Run the Applications
Desktop GUI:
python 5_prediction_interface_fixed.py
Web Interface:

streamlit run 6_web_interface.py

📦 Requirements.txt
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
tensorflow==2.13.0
opencv-python==4.8.0.74
pillow==10.0.0
streamlit==1.25.0
streamlit-drawable-canvas==0.9.3

🎯 Usage Examples
Drawing a Digit
Run python 5_prediction_interface_fixed.py
Draw a digit (0-9) in the canvas
Click "Predict Digit"
View prediction and confidence

Uploading an Image
Run streamlit run 6_web_interface.py
Choose "Upload Image"
Select a digit image
View results with probability chart

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

👨‍💻 Author
Ankit
GitHub: @Ankith-18
Project Repository: Handwritten-Digit-Recognizer


