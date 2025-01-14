import cv2
import numpy as np

def pixel_binning(image, bin_size):
    """
    Perform pixel binning on an image.

    Parameters:
    - image: Input image (grayscale or color).
    - bin_size: The size of the bin (e.g., 2 for 2x2 binning).

    Returns:
    - Binned image with reduced resolution.
    """
    if bin_size <= 1:
        return image  

    # Get image dimensions
    height, width = image.shape[:2]

    # Resize to ensure divisibility by bin_size
    new_height = (height // bin_size) * bin_size
    new_width = (width // bin_size) * bin_size
    image = image[:new_height, :new_width]

    # Reshape and compute binning
    binned = image.reshape(
        new_height // bin_size, bin_size,
        new_width // bin_size, bin_size, -1
    ).mean(axis=(1, 3))  # Average across bin

    # Handle single-channel or multi-channel images
    if len(image.shape) == 3:  # Color image
        binned = binned.astype(np.uint8)
    else:  # Grayscale image
        binned = binned[:, :, 0].astype(np.uint8)

    return binned

# Load an example image
image_path = "assets/noise.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Perform pixel binning with different bin sizes
bin_size_2 = pixel_binning(image, 2)  # 2x2 binning
bin_size_3 = pixel_binning(image, 3)  # 3x3 binning

# Display original and binned images
cv2.imshow("Original Image", image)
cv2.imshow("2x2 Binned Image", cv2.resize(bin_size_2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST))
cv2.imshow("3x3 Binned Image", cv2.resize(bin_size_3, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST))

cv2.waitKey(0)
cv2.destroyAllWindows()
