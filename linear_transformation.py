import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
img = cv2.imread('assets/low_c.jpeg', cv2.IMREAD_COLOR)
rows, cols, _ = img.shape  # Get the dimensions of the image

# Define the rotation matrix for a 45-degree rotation
theta = np.radians(45)   # Convert angle to radians
rotation_matrix = np.array([
    [1, -1],
    [0, 1]
])

# Generate a grid of pixel coordinates
x, y = np.meshgrid(np.arange(cols), np.arange(rows))  # Create a grid of x and y coordinates
coords = np.stack([x.ravel(), y.ravel()], axis=1)  # Flatten and stack into a (N, 2) array

# Center the coordinates (optional, for rotation around the center)
center = np.array([cols / 2, rows / 2])
coords_centered = coords - center

# Apply the transformation
transformed_coords = np.dot(coords_centered, rotation_matrix.T) + center

# Map the transformed coordinates back to the original image
transformed_coords = np.round(transformed_coords).astype(int)
mask = (
    (transformed_coords[:, 0] >= 0) & (transformed_coords[:, 0] < cols) &
    (transformed_coords[:, 1] >= 0) & (transformed_coords[:, 1] < rows)
)
valid_coords = transformed_coords[mask]
original_coords = coords[mask]

# Create a new blank image and map the transformed coordinates
transformed_img = np.zeros_like(img)
transformed_img[valid_coords[:, 1], valid_coords[:, 0]] = img[original_coords[:, 1], original_coords[:, 0]]

# Display the original and transformed images
# cv2.imshow('Original Image', img)
# cv2.imshow('Transformed Image', transformed_img)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('orginal')
plt.subplot(1,2,2)
plt.imshow(transformed_img)
plt.title('transformed')
plt.show()

