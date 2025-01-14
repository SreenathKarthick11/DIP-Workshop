import cv2
import numpy as np

def calculate_brightness(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        return None
    
    # Calculate the average brightness
    brightness = np.mean(image)
    return brightness

# Path to your image
image_path = "assets/bird.png"

# Calculate and display the brightness
brightness = calculate_brightness(image_path)
if brightness is not None:
    print(f"Average Brightness: {brightness:.2f}")
