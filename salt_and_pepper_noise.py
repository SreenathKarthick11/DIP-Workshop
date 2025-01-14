import cv2
import numpy as np
import random

# to create salt and pepper noise on image
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # Convert the image to a numpy array
    img_array = np.array(image)

    # Get the image dimensions
    height, width, _ = img_array.shape
    
    # Generate salt and pepper noise
    total_pixels = height * width
    salt_pixels = int(total_pixels * salt_prob)
    pepper_pixels = int(total_pixels * pepper_prob)
    
    # Add salt (white pixels)
    for _ in range(salt_pixels):
        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)
        img_array[row, col] = [255, 255, 255]  # White pixel for salt noise
    
    # Add pepper (black pixels)
    for _ in range(pepper_pixels):
        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)
        img_array[row, col] = [0, 0, 0]  # Black pixel for pepper noise
    
    return img_array



image = cv2.imread('assets/high_c.jpeg')
noisy_image = add_salt_and_pepper_noise(image, 0.05, 0.05)
cv2.imshow('Noisy Image', noisy_image)

# Apply the mean filter using cv2.blur() # blur take the average of the pixels arround it
mean_filtered_image = cv2.blur(noisy_image, (3, 3))  # (3, 3) is the kernel size

# mean_filtered_image = cv2.boxFilter(image, -1, (5, 5))

# median filter
median_filtered_image = cv2.medianBlur(image,3)  # 5 is the kernel size ( must be odd)

# Display the original and filtered images
cv2.imshow('Mean Filtered Image', mean_filtered_image)
cv2.imshow('Median Filtered Image',median_filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

