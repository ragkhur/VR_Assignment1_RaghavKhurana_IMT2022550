import cv2
import numpy as np
import os

def detect_and_match_features(img1, img2):
    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use FLANN matcher instead of BFMatcher as it's more suitable for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test for better match filtering
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Ratio test
            good_matches.append(m)

    return kp1, kp2, good_matches

def draw_matches(img1, kp1, img2, kp2, matches):
    # Draw only good matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img

def stitch_images(img1, img2):
    # Convert images to grayscale if they aren't already
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    kp1, kp2, matches = detect_and_match_features(gray1, gray2)

    if len(matches) < 4:
        raise Exception("Not enough matches found for homography")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get dimensions
    height, width = img2.shape[:2]
    
    # Warp perspective
    result = cv2.warpPerspective(img1, H, (width * 2, height))
    result[0:height, 0:width] = img2

    return result

def create_panorama(image_paths):
    # Read images
    images = [cv2.imread(path) for path in image_paths]
    
    # Convert to grayscale for feature detection
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Match and draw features between consecutive pairs
    kp1, kp2, matches1 = detect_and_match_features(gray_images[0], gray_images[1])
    kp2, kp3, matches2 = detect_and_match_features(gray_images[1], gray_images[2])

    match_img1 = draw_matches(gray_images[0], kp1, gray_images[1], kp2, matches1)
    match_img2 = draw_matches(gray_images[1], kp2, gray_images[2], kp3, matches2)

    # Stitch images
    pano1 = stitch_images(images[0], images[1])
    panorama = stitch_images(pano1, images[2])

    return match_img1, match_img2, panorama

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, '..', 'images')
result_dir = os.path.join(current_dir, '..', 'results')

# Image paths
image_paths = [
    os.path.join(images_dir, "stitch_1.jpg"),
    os.path.join(images_dir, "stitch_2.jpg"),
    os.path.join(images_dir, "stitch_3.jpg")
]

# Create panorama
match1, match2, panorama = create_panorama(image_paths)

# Save results

cv2.imwrite(os.path.join(result_dir, "final_panorama.jpg"), panorama)

# Display results

cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()