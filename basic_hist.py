import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, color_space="BGR"):
    """
    Plot histograms for an image.

    Parameters:
    - image: Input image.
    - color_space: "BGR" for color images or "GRAY" for grayscale.
    """
    if color_space == "GRAY":
        # Grayscale image histogram
        
        plt.subplot(1,2,1)
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='black') # ravel is used to convert 2D to 1D
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.subplots_adjust(hspace=0.4,wspace=0.4)
        plt.subplot(1,2,2)
        plt.imshow(image,cmap='gray')
        plt.title("Displayed Image")
        plt.axis("off")
        plt.show()
        
    elif color_space == "BGR":
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(2,2,1)
        plt.hist(image[:,:,2].ravel(),bins=256, range=[0, 256], color='r')
        plt.title("red Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        
        plt.subplot(2,2,2)
        plt.hist(image[:,:,1].ravel(),bins=256, range=[0, 256], color='g')
        plt.title("red Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.subplot(2,2,3)
        plt.hist(image[:,:,0].ravel(),bins=256, range=[0, 256], color='b')
        plt.title("blue Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.subplot(2,2,4)
        plt.imshow(image_rgb)
        plt.title("Displayed Image")
        plt.axis("off")  # Hides axis for a cleaner view
        plt.subplots_adjust(hspace=0.5,wspace=0.4)
        plt.show()
# Load an image
image_path = "assets\dog.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Plot histograms
plot_histogram(gray_image, color_space="GRAY")  # Grayscale histogram
plot_histogram(image, color_space="BGR")       # Color histogram
