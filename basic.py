import cv2
import numpy as np
# loading an image
# -1,cv2.IMREAD_COLOR : Load a color image.Any transparency of image will be neglected
# 0,cv2.IMREAD_GREYSCALE : Load image in grayscale mode
# 1,cv2.IMREAD_UNCHANGED : Load image as such including alpha channel
img = cv2.imread('assets/bear.jpg')
print("Image shape:", img.shape) #(rows,cols,no of channels in color)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# for resize image
# img = cv2.resize(img,(400,400)) # (x,y) pixel size
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5) # another way of relative resizing with respect to intial image

# to rotate image
img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
print(img)
# to display image
cv2.imshow("Image",img) # label,image variable
cv2.imshow("grey",gray_image)
# to save an image
# cv2.imwrite('bearg.jpg',img) # name,source

cv2.waitKey(0)  # 0 is for infinite amount of time untill any key is pressed 
                # any other number is for wait for that much seconds
cv2.destroyAllWindows()