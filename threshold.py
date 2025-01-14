import cv2
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv2.imread('assets/bird.png',0)
img= cv2.medianBlur(img,5)
# cv2.THRESH_BINARY: Converts to black-and-white binary.
# cv2.THRESH_BINARY_INV: Inverts binary output.
# cv2.THRESH_TRUNC: Caps pixel intensity at the threshold.
# cv2.THRESH_TOZERO: Keeps values above the threshold, sets others to zero.
# cv2.THRESH_TOZERO_INV: Keeps values below the threshold, sets others to zero.
    
# Apply different thresholding techniques
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

# Create a figure to display all images
plt.figure(figsize=(12, 8))

# Display the original image
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

# Display the Binary Threshold image
plt.subplot(2, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary')
plt.axis('off')

# Display the Binary Inverted Threshold image
plt.subplot(2, 3, 3)
plt.imshow(binary_inv, cmap='gray')
plt.title('Binary Inverted')
plt.axis('off')

# Display the Truncated Threshold image
plt.subplot(2, 3, 4)
plt.imshow(trunc, cmap='gray')
plt.title('Truncated')
plt.axis('off')

# Display the To Zero Threshold image
plt.subplot(2, 3, 5)
plt.imshow(tozero, cmap='gray')
plt.title('To Zero')
plt.axis('off')

# Display the To Zero Inverted Threshold image
plt.subplot(2, 3, 6)
plt.imshow(tozero_inv, cmap='gray')
plt.title('To Zero Inverted')
plt.axis('off')
plt.subplots_adjust(hspace=0.6,wspace=0.4)
# Show the figure
plt.tight_layout()
plt.show()

