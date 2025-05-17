import cv2
import numpy as np

original_image = cv2.imread(r"C:\Users\AC\Pictures\penguin.jpg")  

#1. Read an Image
cv2.imshow("Original Image", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2. Change Color of Top-Right Quarter to light pink
i = original_image.copy()
h, w, _ = i.shape
i[:h//2, w//2:] = [255, 200, 255]  
cv2.imshow("Top-Right Colored (light pink)", i)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Brighten Bottom-Left Quarter
sh = original_image.copy()
sh[h//2:, :w//2] = cv2.add(sh[h//2:, :w//2], 80)  # Increase brightness by 80
cv2.imshow("Bottom-Left Brightened", sh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Image Enhancement Using Histogram Equalization
image = original_image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
cv2.imshow("Enhanced Image", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Remove noise using Gaussian Blur
mm= original_image.copy()
blurred = cv2.GaussianBlur(mm, (5, 5), 0)
cv2.imshow("Noise Removed", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Apply insulation using a threshold.
ss = original_image.copy()
gray = cv2.cvtColor(ss, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholding", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#7. Add a watermark at the top and reduce its size to 4x4 cm
image = original_image.copy()
watermark = cv2.imread("C:\\Users\\AC\\Pictures\\SH.png", cv2.IMREAD_UNCHANGED)

watermark = cv2.cvtColor(watermark, cv2.COLOR_BGRA2BGR)
   
width_cm, height_cm = 4,4 
dpi = 200 
new_w, new_h = int(width_cm * dpi / 2.54), int(height_cm * dpi / 2.54)
watermark_resized = cv2.resize(watermark, (new_w, new_h))
image[:new_h, :new_w] = watermark_resized

cv2.imshow("Watermarked Image (Top, Resized)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8.Foreground Extraction
image = original_image.copy()
mask = cv2.inRange(image, (50, 50, 50), (200, 200, 200))
foreground = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Foreground Extracted", foreground)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 9. Divide the image horizontally into 3 parts and convert them to HSV, Grayscale, and LAB
image = original_image.copy()
h_part = h // 3
hsv = cv2.cvtColor(image[:h_part, :], cv2.COLOR_BGR2HSV)  # Upper part → HSV
gray = cv2.cvtColor(image[h_part:2*h_part, :], cv2.COLOR_BGR2GRAY)  # Middle part → Grayscale
lab = cv2.cvtColor(image[2*h_part:, :], cv2.COLOR_BGR2LAB)  # Bottom → LAB

gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert Grayscale to BGR for proper merging
merged_image = np.vstack((hsv, gray_colored, lab))  

cv2.imshow("Color Space Conversion (HSV, Grayscale, LAB)", merged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




