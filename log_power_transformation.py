import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale (B/W) image
img = cv2.imread('assets/bear.jpg', cv2.IMREAD_GRAYSCALE)

# Log Transformation
c_log = 255 / np.log(1 + np.max(img))  # Normalize
log_transformed = c_log * np.log(1 + img)
log_transformed = np.uint8(log_transformed)

# Power-law (Gamma) Transformation
gamma = 2.2  # Example gamma value (try different values like 0.5, 1.5, 2.0)
c_gamma = 255 / (np.max(img) ** gamma)  # Normalize
gamma_transformed = c_gamma * (img ** gamma)
gamma_transformed = np.uint8(gamma_transformed)

# Plot original and transformed images with histograms
images = [img, log_transformed, gamma_transformed]
titles = ['Original Image', 'Log Transformation', 'Gamma Transformation']

plt.figure(figsize=(10, 6))

for i in range(3):
    # Image
    plt.subplot(3, 2, i * 2 + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

    # Histogram
    plt.subplot(3, 2, i * 2 + 2)
    plt.hist(images[i].ravel(), bins=256, range=(0, 256), color='black')
    plt.title(f'{titles[i]} Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')

plt.tight_layout()
plt.show()
