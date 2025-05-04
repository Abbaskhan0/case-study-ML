
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps 
import os

def preprocess_image(image_path):
    try:
        # Load the image using Pillow
        img = Image.open(image_path)
        display_image = img.copy()  # Copy for visualization

        # Convert to grayscale
        img = img.convert('L')

        # Invert if background is white (MNIST uses black bg)
        if np.mean(np.array(img)) > 128:
            print("Detected light background, inverting image colors for model.")
            img = ImageOps.invert(img)

        # Resize while preserving aspect ratio and fit into a 28x28 canvas
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)
        new_img = Image.new('L', (28, 28), color=0)  # Black canvas
        upper_left_x = (28 - img.width) // 2
        upper_left_y = (28 - img.height) // 2
        new_img.paste(img, (upper_left_x, upper_left_y))

        img_array = np.array(new_img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dim
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dim

        print(f"Image preprocessed successfully. Shape for model: {img_array.shape}")
        return img_array, display_image

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None, None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def predict_digit(image_path, model):
  
    processed_image, display_image = preprocess_image(image_path)
    if processed_image is None:
        return # Error occurred during preprocessing
    try:
        print("Performing prediction...")
        predictions = model.predict(processed_image, verbose=0)
        # predictions is an array of probabilities for each class (0-9)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100 # Confidence percentage

        print(f"\nPrediction complete:")
        print(f"  Predicted Digit: {predicted_digit}")
        print(f"  Confidence: {confidence:.2f}%")
        plt.figure(figsize=(6, 6))
        plt.imshow(display_image, cmap=plt.cm.gray) # Display original image
        plt.title(f"Input Image\nPredicted Digit: {predicted_digit} ({confidence:.1f}%)")
        plt.axis('off') 
        plt.show()
     
        # plt.figure(figsize=(6, 6))
        # plt.imshow(display_image, cmap=plt.cm.gray) # Display original image
        # plt.title(f"Input Image\nPredicted Digit: {predicted_digit} ({confidence:.1f}%)")
        # plt.axis('off') 
        # plt.show()

    except Exception as e:
        print(f"Error during prediction or display: {e}")


