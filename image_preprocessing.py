import cv2
import numpy as np
image = cv2.imread('61VUeSd9PXL._SY500.jpg')
        
resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Apply a large Gaussian blur to estimate the background
blur = cv2.GaussianBlur(gray, (51, 51), 0)
# Subtract the background
background_subtracted = cv2.subtract(gray, blur)
 
normalized = cv2.normalize(background_subtracted, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(normalized)

# Estimate the illumination
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
illumination = cv2.morphologyEx(clahe_result, cv2.MORPH_CLOSE, kernel)
# Correct the image by dividing by the illumination
illumination_corrected = cv2.divide(clahe_result, illumination, scale=255)

_, binarized = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

binarized = cv2.GaussianBlur(binarized, (5, 5), 0)

cv2.imwrite('binarized.jpg', binarized)

kernel = np.ones((5,5), np.uint8)   # 5x5 kernel
opening = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
dilation = cv2.dilate(binarized, kernel, iterations=1)

cv2.imwrite('opening.jpg', opening)
cv2.imwrite('dilation.jpg', dilation)

lines = cv2.HoughLinesP(binarized, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Create a copy to visualize lines
line_image = np.copy(resized)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the result for visualization
cv2.imwrite('hough_lines.jpg', line_image)
