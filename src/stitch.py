import numpy as np
import cv2
import imutils
import os
import matplotlib.pyplot as plt

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Input and output directories (same as your previous version)
images_dir = os.path.join(current_dir, '..', 'images')
results_dir = os.path.join(current_dir, '..', 'results')

# Ensure results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load and sort image files
files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
images = []

# Load images and display them
for file in files:
    pathToImage = os.path.join(images_dir, file)
    img = cv2.imread(pathToImage)
    
    if img is None:
        print(f"[ERROR] Could not load image: {file}")
        continue  # Skip invalid images
    
    images.append(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(f"Image: {file}")
    plt.axis("off")
    plt.show()

# Stitch images together
imageStitcher = cv2.Stitcher_create()
error, stitchedImage = imageStitcher.stitch(images)

if error == cv2.Stitcher_OK:
    # Save stitched output before processing
    stitched_path = os.path.join(results_dir, "stitchedOutput.png")
    cv2.imwrite(stitched_path, stitchedImage)

    # Resize for display
    stitchedResized = cv2.resize(stitchedImage, (800, 600))
    
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(stitchedResized, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Resized Image")
    plt.axis("off")
    plt.show()

    # Add a border
    stitchedImage = cv2.copyMakeBorder(stitchedImage, 5, 5, 5, 5, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2GRAY)
    threshImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Find contours
    contours = cv2.findContours(threshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        print("[ERROR] No contours found!")
    else:
        areaOI = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(areaOI)

        # Crop the stitched image
        stitchedImage = stitchedImage[y:y + h, x:x + w]

        # Save final processed output
        final_output_path = os.path.join(results_dir, "stitchedOutputProcessed.png")
        cv2.imwrite(final_output_path, stitchedImage)

        # Show final output
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2RGB))
        plt.title("Final Stitched Image")
        plt.axis("off")
        plt.show()

else:
    print("[ERROR] Stitching failed!")

cv2.waitKey(0)
cv2.destroyAllWindows()
