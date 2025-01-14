import cv2
import numpy as np
import matplotlib.pyplot as plt

def double_threshold(image, lower_thresh, upper_thresh):
    # Step 1: Apply lower threshold
    lower_thresholded = np.where(image >= lower_thresh, image, 0)

    # Step 2: Apply higher threshold on the result of the lower threshold
    higher_thresholded = np.where(lower_thresholded >= upper_thresh, 255, 0)

    # Convert to uint8 for visualization
    return higher_thresholded.astype(np.uint8)

# Read and preprocess the image
img = cv2.imread('assets/coins.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection with double thresholding
lower_threshold = 50
upper_threshold = 150
edges = cv2.Canny(img, lower_threshold, upper_threshold)
result= double_threshold(img,lower_threshold,upper_threshold)
# Display the results
plt.figure(figsize=(8, 8))
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(1, 3, 2), plt.imshow(edges, cmap='gray')
plt.title('Canny Edges Double Thresholding)'), plt.axis('off')

plt.subplot(1, 3, 3), plt.imshow(result, cmap='gray')
plt.title('Double Thresholded Image'), plt.axis('off')

plt.show()
