import cv2  

img = cv2.imread('assets/lowlight.jpg', cv2.IMREAD_COLOR)

# Step 2: Convert the image to YCrCb color space
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# YCrCb separates the brightness (Y) channel from color information (Cr, Cb).
# This allows us to process only the luminance (Y) channel for contrast enhancement.

# Step 3: Create a CLAHE object for local contrast enhancement [Contrast-limited adaptive histogram equalization (CLAHE)]
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
# 'clipLimit' sets the threshold for contrast limiting. Higher values increase contrast.
# 'tileGridSize' divides the image into 8x8 tiles for local processing.

# Step 4: Apply CLAHE to the Y (luminance) channel
ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
# CLAHE is applied only to the Y channel to enhance brightness and contrast.
# The Cr and Cb (chrominance) channels remain unchanged to preserve color fidelity.

# Step 5: Convert the processed image back to BGR color space
result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
# Converts the enhanced YCrCb image back to BGR format for display.


cv2.imshow('original', img)  
cv2.imshow('result', result)  

cv2.waitKey(0) 
cv2.destroyAllWindows()  
