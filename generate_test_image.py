# generate_test_image.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

def generate_multi_digit_image(digits_str="12345", filename=None):
    """
    Generate a test image with multiple digits
    
    Args:
        digits_str: String of digits to draw (e.g., "12345")
        filename: Output filename (optional)
    
    Returns:
        PIL Image object
    """
    # Create blank image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font
    try:
        # Try different font paths for different systems
        font_paths = [
            "arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/Helvetica.ttc"  # Mac
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 48)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw digits with slight variations
    x_offset = 20
    for i, digit in enumerate(digits_str):
        # Add slight random variation to position
        y_offset = 20 + random.randint(-3, 3)
        
        # Draw the digit
        draw.text((x_offset, y_offset), digit, fill='black', font=font)
        x_offset += 45 + random.randint(-2, 2)
    
    # Convert to numpy array for adding noise
    img_np = np.array(img)
    
    # Add some random noise for realism
    noise = np.random.normal(0, 15, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    
    # Add slight blur
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0.5)
    
    # Convert back to PIL
    result_img = Image.fromarray(img_np)
    
    # Save if filename provided
    if filename:
        result_img.save(filename)
        print(f" Generated: {filename}")
    
    return result_img

def generate_phone_number_image():
    """Generate a phone number style image"""
    phone = f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
    return generate_multi_digit_image(phone.replace("-", ""), f"test_phone_{phone}.png")

def generate_zipcode_image():
    """Generate a ZIP code style image"""
    zipcode = f"{random.randint(10000,99999)}"
    return generate_multi_digit_image(zipcode, f"test_zip_{zipcode}.png")

def generate_price_image():
    """Generate a price tag style image"""
    price = f"${random.randint(10,99)}.{random.randint(0,99):02d}"
    img = generate_multi_digit_image(price.replace("$", "").replace(".", ""), f"test_price_{price}.png")
    return img

if __name__ == "__main__":
    print(" Generating test images for multi-digit recognition...")
    print("="*50)
    
    # Create output directory if it doesn't exist
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
        print(" Created 'test_images' folder")
    
    # Generate various test images
    test_cases = [
        "12345",
        "67890",
        "55555",
        "2024",
        "9876",
        "100200",
        "314159",  # Pi digits
        "271828",  # e digits
    ]
    
    for digits in test_cases:
        filename = f"test_images/test_digits_{digits}.png"
        generate_multi_digit_image(digits, filename)
    
    # Generate specialized images
    for _ in range(3):
        generate_phone_number_image()
        generate_zipcode_image()
        generate_price_image()
    
    print("="*50)
    print(" All test images generated successfully!")
    print(f" Check the 'test_images' folder")
    print("\n Next steps:")
    print("1. Run: streamlit run 7_advanced_multi_digit_recognizer.py")
    print("2. Upload images from the 'test_images' folder")
    print("3. See multi-digit recognition in action!")