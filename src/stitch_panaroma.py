import cv2
import numpy as np
import os

def detect_and_match_features(img1, img2):
    #key points are detected using ORB and BFMatcher
    orb = cv2.ORB_create()

    #detecting keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches

def draw_matches(img1, kp1, img2, kp2, matches):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return match_img

def stitch_images(img1, img2):
    kp1, kp2, matches = detect_and_match_features(img1, img2)

    scr_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(scr_pts, dst_pts, cv2.RANSAC)

    height, width = img2.shape[:2]
    result = cv2.warpPerspective(img1, H, (width*2, height))
    result[0:height, 0:width] = img2

    return result

def create_panaroma(image_paths):

    images = [cv2.imread(path) for path in image_paths]

    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    kp1, kp2, matches1 = detect_and_match_features(gray_images[0], gray_images[1])
    kp2, kp3, matches2 = detect_and_match_features(gray_images[1], gray_images[2])

    match_img1 = draw_matches(gray_images[0], kp1, gray_images[1], kp2, matches1)
    match_img2 = draw_matches(gray_images[1], kp2, gray_images[2], kp3, matches2)
    pano1 = stitch_images(images[0], images[1])
    panorama = stitch_images(pano1, images[2])

    return match_img1, match_img2, panorama

current_dir = os.path.dirname(os.path.abspath(__file__))

images_dir = os.path.join(current_dir, '..', 'images')
result_dir = os.path.join(current_dir, '..', 'results')

image_paths = [
    os.path.join(images_dir, "stitch_1.jpg"),
    os.path.join(images_dir, "stitch_2.jpg"),
    os.path.join(images_dir, "stitch_3.jpg")
]

match1, match2, panorama = create_panaroma(image_paths)

cv2.imwrite(os.path.join(result_dir, "matches_1_2.jpg"), match1)
cv2.imwrite(os.path.join(result_dir, "matches_2_3.jpg"), match2)
cv2.imwrite(os.path.join(result_dir, "final_panaroma.jpg"), panorama)

cv2.imshow("Matches 1-2", match1)
cv2.imshow("Matches 2-3", match2)
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

