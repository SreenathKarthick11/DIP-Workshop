import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(img, s_min=0, s_max=255):
    # Find the minimum and maximum intensity values in the input image
    r_min, r_max = np.min(img), np.max(img)
    
    # Perform contrast stretching
    stretched = (img - r_min) / (r_max - r_min) * (s_max - s_min) + s_min
    
    # Convert to uint8 format for visualization
    return np.uint8(stretched)

# Read the grayscale image
img = cv2.imread('assets/trees.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "Image not found!"

# Apply contrast stretching
stretched_img = contrast_stretching(img)

# Plot original and stretched images with histograms
plt.figure(figsize=(12, 8))

# Original Image and Histogram
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0, 256), color='black')
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# Contrast Stretched Image and Histogram
plt.subplot(2, 2, 3)
plt.imshow(stretched_img, cmap='gray')
plt.title("Contrast Stretched Image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(stretched_img.ravel(), bins=256, range=(0, 256), color='black')
plt.title("Stretched Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
