import cv2
import numpy as np
import matplotlib.pyplot as plt

## example of addition
def add_example():
    # Load the images
    img1 = cv2.imread('assets/towers.jpg')  # Image 1 (Base image)
    img2 = cv2.imread('assets/plane.png', cv2.IMREAD_UNCHANGED)  # Image 2 (with alpha channel)

    # Resize img2 to match the size of img1 if necessary
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    # Separate the color channels and the alpha channel from img2
    bgr_img2 = img2_resized[:, :, :3]  # Extract the BGR channels
    alpha_channel = img2_resized[:, :, 3]  # Extract the alpha channel

    # Normalize the alpha channel to [0, 1] range
    alpha = alpha_channel / 255.0

    # Perform the alpha blending
    blended = np.zeros_like(img1, dtype=np.float32)

    # Iterate through each pixel and blend the images
    for c in range(0, 3):  # Iterate over the BGR channels
        blended[:, :, c] = (1 - alpha) * img1[:, :, c] + alpha * bgr_img2[:, :, c]

    # Convert the result back to uint8
    blended = np.uint8(blended)

    # Show the original and blended images
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title('Result')
    plt.show()

def sub_example():
    img1 = cv2.imread('assets/cat.jpg',0)  # Image 1 (Base image)
    img2 = cv2.imread('assets/cat2.jpg',0)
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0])) 
    subtracted_img = img1 - img2_resized

    # Clip values to stay within valid image range (0-255)
    #subtracted_img = np.clip(subtracted_img, 0, 255)

    # Show the result
    plt.subplot(3,1,1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.subplot(3,1,2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.subplot(3,1,3)
    plt.imshow(subtracted_img, cmap='gray')
    plt.title('Result')
    plt.subplots_adjust(hspace=0.6,wspace=0.4)
    plt.show()

def logical_example():
    img1 = cv2.imread('assets/cat.jpg',1)  # Image 1 (Base image)
    img2 = cv2.imread('assets/cat2.jpg',1)

    plt.subplot(2,2,1)
    plt.imshow(cv2.bitwise_and(img1, img2))
    plt.title('And')
    plt.subplot(2,2,2)
    plt.imshow(cv2.bitwise_or(img1,img2))
    plt.title('Or')
    plt.subplot(2,2,3)
    plt.imshow(cv2.bitwise_not(img1,img2))
    plt.title('Not')
    plt.subplot(2,2,4)
    plt.imshow(cv2.bitwise_xor(img1,img2))
    plt.title('Xor')
    plt.subplots_adjust(hspace=0.6,wspace=0.4)
    plt.show()

logical_example()