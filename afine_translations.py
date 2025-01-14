import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
img = cv2.imread('assets/bear.jpg')
rows, cols, _ = img.shape

# 1. Scaling
scale_matrix = np.array([
    [1.5, 0, 0],  # Scale x by 1.5
    [0, 1.5, 0]   # Scale y by 1.5
], dtype=np.float32)
scaled_img = cv2.warpAffine(img, scale_matrix, (cols, rows))

# 2. Rotation (45 degrees)
theta = np.radians(45)
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0]
], dtype=np.float32)
rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

# 3. Shearing
shear_matrix = np.array([
    [1, 0.5, 0],  # Shear along x
    [0.5, 1, 0]   # Shear along y
], dtype=np.float32)
sheared_img = cv2.warpAffine(img, shear_matrix, (cols, rows))

# 4. Translation
translation_matrix = np.array([
    [1, 0, 100],  # Translate x by 100 pixels
    [0, 1, 50]    # Translate y by 50 pixels
], dtype=np.float32)
translated_img = cv2.warpAffine(img, translation_matrix, (cols, rows))

# reflection
reflected_matrix = np.array([
    [-1,0,cols],           # [1,0,0],[0,-1,rows]
    [0,1,0]
], dtype=np.float32)
reflected_img = cv2.warpAffine(img,reflected_matrix,(cols,rows))
# Display results

plt.subplot(2,3,1)
plt.imshow(img)
plt.title('orginal')
plt.subplot(2,3,2)
plt.imshow(scaled_img)
plt.title('scaled')
plt.subplot(2,3,3)
plt.imshow(rotated_img)
plt.title('rotated')
plt.subplot(2,3,4)
plt.imshow(sheared_img)
plt.title('sheared')
plt.subplot(2,3,5)
plt.imshow(translated_img)
plt.title('translated')
plt.subplot(2,3,6)
plt.imshow(reflected_img)
plt.title('reflected')
plt.show()

