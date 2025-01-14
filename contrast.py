import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_plot_hist(img1,img2):
    plt.subplot(2,2,1)
    plt.hist(img1.ravel(), bins=256, range=[0, 256], color='black') # ravel is used to convert 2D to 1D
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.subplot(2,2,2)
    plt.imshow(img1,cmap='gray')
    plt.title("low contrast Image")
    plt.axis("off")
    plt.subplot(2,2,3)
    plt.hist(img2.ravel(), bins=256, range=[0, 256], color='black') # ravel is used to convert 2D to 1D
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.subplot(2,2,4)
    plt.imshow(img2,cmap='gray')
    plt.title("high contrast Image")
    plt.axis("off")
    plt.subplots_adjust(hspace=0.6,wspace=0.4)
    plt.show()

low_contrast_image=cv2.imread('assets/land_l.png')
high_contrast_image=cv2.imread('assets/land_h.png')

contrast_plot_hist(low_contrast_image,high_contrast_image)