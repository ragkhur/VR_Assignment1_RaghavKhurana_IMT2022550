import cv2
import os
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current .py file

# Paths to input and output directories
images_dir = os.path.join(current_dir, '..', 'images')
results_dir = os.path.join(current_dir, '..', 'results')

# Input image filename
input_image_name = "coins2.jpeg"
input_image_path = os.path.join(images_dir, input_image_name)

# 2. Load the image
image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
if image is None:
    print(f"Error: Could not load image {input_image_path}")
    exit()

# 3. Convert image to grayscale and apply Gaussian blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# 4. Otsu's thresholding (inverted so coins appear white)
ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 5. Morphological opening to remove small noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# 6. Dilation to get sure background
sure_bg = cv2.dilate(opening, kernel, iterations=2)

# 7. Distance transform for sure foreground
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# 8. Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# 9. Connected components for markers
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 10. Apply Watershed
markers = cv2.watershed(image, markers)

# Create a copy for visualization
segmented_image = image.copy()
segmented_image[markers == -1] = [0, 0, 255]

# 11. Save the overall segmented image
segmented_image_name = "Segmented_coins.jpg"
segmented_image_path = os.path.join(results_dir, segmented_image_name)
cv2.imwrite(segmented_image_path, segmented_image)

# 12. Extract and save each coin individually
unique_markers = np.unique(markers)
coin_count = 0
for marker in unique_markers:
    if marker <= 1:
        continue  # Skip background (1) and boundary (-1)
    # Create a mask for this coin
    coin_mask = np.zeros_like(gray, dtype="uint8")
    coin_mask[markers == marker] = 255

    # Find contours
    contours, _ = cv2.findContours(coin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        coin_roi = image[y:y+h, x:x+w]

        coin_count += 1
        coin_filename = f"segmented_coin_{coin_count}.jpg"
        coin_path = os.path.join(results_dir, coin_filename)
        cv2.imwrite(coin_path, coin_roi)

# 13. Show result (optional)
cv2.imshow("Segmented Coins", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(coin_count)
exit()
