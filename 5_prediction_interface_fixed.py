# 5_prediction_interface_fixed.py
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageTk
import os

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("800x600")
        
        # Create GUI elements first
        self.create_widgets()
        
        # Then load model
        self.load_model()
        
        # Drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create a blank image for drawing
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Handwritten Digit Recognizer",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Canvas for drawing
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack()
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=280,
            height=280,
            bg='white',
            cursor='cross'
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_on_canvas)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, pady=10)
        button_frame.pack()
        
        # Predict button
        self.predict_btn = tk.Button(
            button_frame,
            text="Predict Digit",
            command=self.predict_digit,
            bg="green",
            fg="white",
            padx=20,
            pady=5,
            font=("Arial", 10, "bold")
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="Clear Canvas",
            command=self.clear_canvas,
            bg="red",
            fg="white",
            padx=20,
            pady=5,
            font=("Arial", 10, "bold")
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Upload button
        upload_btn = tk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_image,
            bg="blue",
            fg="white",
            padx=20,
            pady=5,
            font=("Arial", 10, "bold")
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = tk.Frame(main_frame, pady=10)
        result_frame.pack()
        
        # Prediction label
        self.prediction_label = tk.Label(
            result_frame,
            text="Draw a digit or upload an image",
            font=("Arial", 14)
        )
        self.prediction_label.pack()
        
        # Confidence label
        self.confidence_label = tk.Label(
            result_frame,
            text="",
            font=("Arial", 12)
        )
        self.confidence_label.pack()
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9)
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
    
    def load_model(self):
        """Load the trained CNN model"""
        possible_models = [
            'best_cnn_model.h5',
            'cnn_digit_recognizer.h5',
            'mlp_digit_recognizer.h5'
        ]
        
        model_loaded = False
        for model_path in possible_models:
            if os.path.exists(model_path):
                try:
                    self.model = load_model(model_path)
                    self.status_label.config(text=f"✅ Model loaded: {model_path}")
                    print(f"Loaded model: {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue
        
        if not model_loaded:
            self.model = None
            self.status_label.config(text="❌ No model found! Please train a model first.")
            self.predict_btn.config(state=tk.DISABLED)
            messagebox.showwarning("Warning", "No model found! Please train a model first.")
    
    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_on_canvas(self, event):
        if self.drawing:
            x, y = event.x, event.y
            
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=15,
                fill='black',
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=0,
                width=15
            )
            
            self.last_x = x
            self.last_y = y
    
    def stop_draw(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        """Clear both canvas and PIL image"""
        self.canvas.delete("all")
        # Reset PIL image
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit or upload an image")
        self.confidence_label.config(text="")
    
    def get_image_array(self):
        """Convert PIL image to array for prediction"""
        # Resize to 28x28
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        
        # Invert colors
        img_array = 1.0 - img_array
        
        # Reshape for model
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict_digit(self):
        """Predict the drawn digit"""
        if self.model is None:
            messagebox.showerror("Error", "No model loaded! Please train a model first.")
            return
        
        try:
            # Get image array
            img_array = self.get_image_array()
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit] * 100
            
            # Update labels
            self.prediction_label.config(
                text=f"Predicted Digit: {predicted_digit}",
                fg="green"
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}%"
            )
            
            # Show probabilities
            self.show_probabilities(predictions)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def show_probabilities(self, predictions):
        """Show probability for each digit"""
        prob_window = tk.Toplevel(self.root)
        prob_window.title("Prediction Probabilities")
        prob_window.geometry("400x350")
        
        # Title
        tk.Label(
            prob_window,
            text="Probability Distribution",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # Create frame for probabilities
        frame = tk.Frame(prob_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for i, prob in enumerate(predictions):
            row_frame = tk.Frame(frame)
            row_frame.pack(fill=tk.X, pady=3)
            
            # Digit label
            digit_label = tk.Label(
                row_frame, 
                text=f"Digit {i}:", 
                width=8, 
                anchor='w',
                font=("Arial", 10)
            )
            digit_label.pack(side=tk.LEFT)
            
            # Probability
            prob_percent = prob * 100
            prob_label = tk.Label(
                row_frame, 
                text=f"{prob_percent:.2f}%", 
                width=8,
                font=("Arial", 10, "bold")
            )
            prob_label.pack(side=tk.LEFT)
            
            # Progress bar
            bar_frame = tk.Frame(row_frame, height=20, width=200, bg='lightgray')
            bar_frame.pack(side=tk.LEFT, padx=5)
            bar_frame.pack_propagate(False)
            
            # Fill bar
            bar_width = int(prob_percent * 2)
            color = 'green' if i == np.argmax(predictions) else 'blue'
            
            bar = tk.Frame(bar_frame, bg=color, width=bar_width, height=20)
            bar.pack(side=tk.LEFT)
            bar.pack_propagate(False)
        
        # Close button
        tk.Button(
            prob_window,
            text="Close",
            command=prob_window.destroy,
            bg="gray",
            fg="white",
            padx=20,
            pady=5
        ).pack(pady=10)
    
    def upload_image(self):
        """Upload and predict from image file"""
        if self.model is None:
            messagebox.showerror("Error", "No model loaded!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                img = Image.open(file_path).convert('L')
                
                # Clear and display
                self.clear_canvas()
                
                # Resize for display
                display_img = img.resize((280, 280), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(display_img)
                self.canvas.create_image(140, 140, image=self.photo)
                
                # Update PIL image
                self.image = display_img.copy()
                self.draw = ImageDraw.Draw(self.image)
                
                # Preprocess and predict
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img)
                img_array = img_array.astype('float32') / 255.0
                img_array = 1.0 - img_array
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predict
                predictions = self.model.predict(img_array, verbose=0)[0]
                predicted_digit = np.argmax(predictions)
                confidence = predictions[predicted_digit] * 100
                
                self.prediction_label.config(
                    text=f"Predicted Digit: {predicted_digit}",
                    fg="green"
                )
                self.confidence_label.config(
                    text=f"Confidence: {confidence:.2f}%"
                )
                
                self.show_probabilities(predictions)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()