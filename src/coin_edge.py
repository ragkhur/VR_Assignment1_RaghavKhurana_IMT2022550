import cv2
import numpy as np
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

images_dir = os.path.join(current_dir, '..', 'images')
results_dir = os.path.join(current_dir, '..', 'results')

input_image_name = "coins1.jpg"
input_image_path = os.path.join(images_dir, input_image_name)

output_image_name = "coints_with_edge_detection.jpeg"
output_image_path = os.path.join(results_dir, output_image_name)

image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
if image is None:
    print(f"Error: Could not load image {input_image_path}")
    exit()


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred=cv2.GaussianBlur(gray, (9,9), 2)
v = np.median(blurred)
lower = int(max(0, 0.66 * v))
upper = int(min(255, 1.33 * v))
edges = cv2.Canny(blurred, lower, upper)


circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1,  
    minDist=60,    # Increase minDist to avoid multiple detections per coin
    param1=55,    # Canny edge high threshold
    param2=37,     # Increase threshold to avoid detecting weak edges
    minRadius=90,  
    maxRadius=140   
)

coin_count = 0
if circles is not None:
    circles= np.uint16(np.around(circles))
    coin_count = circles.shape[1]
    for i in circles[0,:]:
        cv2.circle(image, (i[0], i[1]), i[2], (0,255,0),2)
        cv2.circle(image, (i[0], i[1]), 2, (0,0,255), 3)


cv2.imwrite(output_image_path, image)

cv2.imshow("Detected coins", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(coin_count)
exit()