import cv2
import numpy as np
import matplotlib.pyplot as plt

def intensity_slicing(img, r_min, r_max, low_value=0):
    sliced_img = np.zeros_like(img, dtype=np.uint8)
     # Range Highlighting
    sliced_img = np.copy(img)
    sliced_img[(img < r_min) | (img > r_max)] = low_value
    
    return sliced_img
r_min, r_max = 50, 250

for i in range(1, 7):
    image_filename = f'assets/w{i}.jpeg'
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5) 
    img = intensity_slicing(img,r_min,r_max)
    if img is not None:

        cv2.imshow(f'Image w{i}', img)
    else:
        print(f"Error: Image {image_filename} not found or could not be loaded.")

cv2.waitKey(0)
cv2.destroyAllWindows()