import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=50):
    # Generate Gaussian noise
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    
    noisy_image = np.clip(image + gauss, 0, 255)  # Ensure values are within valid pixel range
    
    return noisy_image.astype(np.uint8)


def geometric_mean_filter(image, kernel_size):
    # Get the image dimensions
    height, width, channels = image.shape
    
    # Create an output image to store the filtered result
    output_image = np.copy(image)
    
    # Calculate the offset for the kernel (half the kernel size)
    offset = kernel_size // 2
    
    # Iterate over every pixel in the image (except borders)
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            # Extract the kernel around the current pixel
            kernel = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            
            # Flatten the kernel to a 1D array and convert to float for precision
            flattened_kernel = kernel.flatten().astype(np.float64)
            
            # Compute the product of the pixel values
            product = np.prod(flattened_kernel)
            
            # Compute the geometric mean
            geom_mean = product ** (1.0 / len(flattened_kernel))
            
            # Assign the geometric mean value to the center pixel
            output_image[i, j] = geom_mean
    
    return output_image.astype(np.uint8)


image=cv2.imread('assets/bird.png')
noisy_image=add_gaussian_noise(image)
cv2.imshow('OG',noisy_image)
#cv2.imshow('Noisy Image',noisy_image)
mean_filtered_image=cv2.blur(noisy_image, (5, 5))
#cv2.imshow('mean filtered image',mean_filtered_image)
GM_filtered_image=geometric_mean_filter(noisy_image,1)
#cv2.imshow('GM filtered Image',GM_filtered_image)
#cv2.imshow('fn',geometric_mean_filter(mean_filtered_image,2))
cv2.imshow('gaussian',cv2.GaussianBlur(noisy_image,(3,3),1))
cv2.waitKey(0)
cv2.destroyAllWindows()