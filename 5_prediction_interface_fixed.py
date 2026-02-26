
# enhanced_digit_recognizer.py
import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import os

class EnhancedDigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer - Write ONE digit at a time!")
        self.root.geometry("500x600")
        
        # Load model
        self.load_model()
        
        # Instructions
        instructions = tk.Label(
            root, 
            text="⚠️ IMPORTANT: Write ONLY ONE digit at a time!\nClear canvas between digits!",
            font=("Arial", 10, "bold"),
            fg="red",
            bg="yellow",
            pady=5
        )
        instructions.pack(fill=tk.X)
        
        # Drawing area
        canvas_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        canvas_frame.pack(pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame, 
            width=280, 
            height=280, 
            bg='white',
            cursor='pencil'
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        
        # PIL image for processing
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        
        # Button frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        # Predict button
        self.predict_btn = tk.Button(
            btn_frame,
            text="🔍 Predict",
            command=self.predict,
            bg="green",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=5
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_btn = tk.Button(
            btn_frame,
            text="🗑️ Clear",
            command=self.clear,
            bg="red",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=5
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Upload button
        upload_btn = tk.Button(
            btn_frame,
            text="📁 Upload",
            command=self.upload_image,
            bg="blue",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=5
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        result_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.result_label = tk.Label(
            result_frame,
            text="Draw a digit (0-9) and click Predict",
            font=("Arial", 14)
        )
        self.result_label.pack()
        
        self.confidence_label = tk.Label(
            result_frame,
            text="",
            font=("Arial", 12)
        )
        self.confidence_label.pack()
        
        # Status bar
        self.status_label = tk.Label(
            root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)
    
    def load_model(self):
        """Load the trained model"""
        model_files = ['best_cnn_model.h5', 'cnn_digit_recognizer.h5']
        for model_file in model_files:
            if os.path.exists(model_file):
                self.model = load_model(model_file)
                self.status_label.config(text=f"Model loaded: {model_file}")
                return
        self.model = None
        self.status_label.config(text="No model found!")
    
    def paint(self, event):
        """Draw on canvas"""
        if self.last_x and self.last_y:
            # Draw on tkinter canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=20, fill='black', capstyle=tk.ROUND, smooth=True
            )
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=0, width=20
            )
        self.last_x, self.last_y = event.x, event.y
    
    def reset(self, event):
        """Reset drawing coordinates"""
        self.last_x, self.last_y = None, None
    
    def clear(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit (0-9) and click Predict")
        self.confidence_label.config(text="")
        self.status_label.config(text="Canvas cleared")
    
    def preprocess_image(self):
        """Convert drawing to model input"""
        # Resize to 28x28
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        
        # Invert (MNIST has white on black)
        img_array = 1.0 - img_array
        
        # Reshape for model
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict(self):
        """Predict the drawn digit"""
        if self.model is None:
            self.result_label.config(text="No model loaded!")
            return
        
        # Preprocess
        img_array = self.preprocess_image()
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        digit = np.argmax(predictions)
        confidence = predictions[digit] * 100
        
        # Show result
        self.result_label.config(
            text=f"Predicted Digit: {digit}",
            fg="green"
        )
        self.confidence_label.config(
            text=f"Confidence: {confidence:.2f}%"
        )
        
        # Show top 3 predictions
        top_3 = np.argsort(predictions)[-3:][::-1]
        top_text = "Top predictions: "
        for d in top_3:
            top_text += f"{d} ({predictions[d]*100:.1f}%)  "
        
        self.status_label.config(text=top_text)
        
        # Warn if multiple digits detected
        if len(np.where(predictions > 0.1)[0]) > 1:
            self.status_label.config(
                text="⚠️ Multiple digits detected! Write only ONE digit at a time.",
                fg="red"
            )
    
    def upload_image(self):
        """Upload an image file"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            # Load and display image
            img = Image.open(file_path).convert('L')
            img.thumbnail((280, 280))
            
            # Clear and display
            self.clear()
            
            # Convert to PhotoImage and display
            from PIL import ImageTk
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(140, 140, image=self.photo)
            
            # Update PIL image
            self.image = img.resize((280, 280))
            self.draw = ImageDraw.Draw(self.image)
            
            # Auto-predict
            self.predict()

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedDigitRecognizer(root)
    root.mainloop()