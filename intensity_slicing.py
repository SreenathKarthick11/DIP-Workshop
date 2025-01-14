import cv2
import numpy as np
import matplotlib.pyplot as plt

def intensity_slicing(img, r_min, r_max, high_value=255, low_value=0, binary=True):
    """
    Apply intensity-level slicing to an image.
    
    Parameters:
        img: Input grayscale image.
        r_min: Lower bound of intensity range.
        r_max: Upper bound of intensity range.
        high_value: Output intensity for pixels within range.
        low_value: Output intensity for pixels outside range.
        binary: If True, performs binary slicing. Otherwise, highlights range.
        
    Returns:
        Transformed image.
    """
    sliced_img = np.zeros_like(img, dtype=np.uint8)
    
    if binary:
        # Binary Slicing
        sliced_img[(img >= r_min) & (img <= r_max)] = high_value
    else:
        # Range Highlighting
        sliced_img = np.copy(img)
        sliced_img[(img < r_min) | (img > r_max)] = low_value
    
    return sliced_img

# Load the grayscale image
img = cv2.imread('assets/bear.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "Image not found!"

# Apply intensity-level slicing
r_min, r_max = 100, 200  # Specify intensity range
binary_sliced = intensity_slicing(img, r_min, r_max, binary=True)
highlighted = intensity_slicing(img, r_min, r_max, binary=False)

# Plot original and transformed images
plt.figure(figsize=(10, 6))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Binary Sliced Image
plt.subplot(2, 2, 2)
plt.imshow(binary_sliced, cmap='gray')
plt.title(f"Binary Slicing (Range: {r_min}-{r_max})")
plt.axis('off')

# Highlighted Range Image
plt.subplot(2, 2, 3)
plt.imshow(highlighted, cmap='gray')
plt.title(f"Range Highlighting (Range: {r_min}-{r_max})")
plt.axis('off')

# Histogram of Original Image
plt.subplot(2, 2, 4)
plt.hist(img.ravel(), bins=256, range=(0, 256), color='black')
plt.title("Histogram of Original Image")
plt.xlabel("Intensity Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
