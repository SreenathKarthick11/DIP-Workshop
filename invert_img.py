import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('assets/dog.jpg')
bw_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

inverted_img = 255 - bw_image
negative_img = 255 - image

plt.subplot(2,2,1)
plt.imshow(bw_image,cmap='gray')
plt.title('Black/White image')
plt.subplot(2,2,2)
plt.imshow(inverted_img,cmap='gray')
plt.title('Inverted Black/White image')
plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal image')
plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB))
plt.title('Negative image')
plt.subplots_adjust(hspace=0.6,wspace=0.4)
plt.show()

