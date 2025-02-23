# VR_Assignment1_RaghavKhurana_IMT2022550

### Raghav Khurana - IMT2022550 

## 📌 Overview  
This repository contains solutions for Computer Vision Assignment 1, which consists of two main tasks:  
1. *Coin Detection and Segmentation* –  Detecting, segmenting, and counting coins in an image using edge detection and watershed segmentation techniques.  
2. *Image Stitching* – Creating a stitched panorama from multiple overlapping images using feature matching and homography estimation.  

All implementations are in *Python*.  

---

## 📂 Repository Structure  


VR_Assignment1_Himanshu_IMT2022584/
│── Q1/                                         # Folder for Coin Detection and Segmentation
│   │── coin_edge.py                           # Python script for coin edge detection and counting
│   │── coin_segmentation.py                   # Python script for coin segmentation using watershed
│   │── coins1.jpg                             # Input image for edge detection
│   │── coins2.jpeg                            # Input image for segmentation
│   │── coints_with_edge_detection.jpeg        # Output image with detected coins highlighted
│   │── Segmented_coins.jpg                    # Output segmented coins image
│   │── segmented_coin_*.jpg                    # Individual segmented coin images
│
│── Q2/                                         # Folder for Image Stitching
│   │── stitch_panaroma.py                     # Python script for panorama stitching
│   │── stitch_1.jpg                           # Input image 1 for stitching
│   │── stitch_2.jpg                           # Input image 2 for stitching
│   │── stitch_3.jpg                           # Input image 3 for stitching
│   │── final_panorama.jpg                     # Output stitched panorama image
│
│── README.md                                   # Project documentation
│── requirements.txt                            # Python package dependencies
│── VR_Assignment1_Himanshu_IMT2584.pdf          # Final report in PDF format



---

## 🛠 Dependencies  
Ensure the following Python packages are installed before running the scripts:  

bash
pip install opencv-python numpy



---

## 🔧 Execution Instructions  
Run each script separately from the terminal or command prompt.  

### *⿡ Coin Detection and Segmentation*  
Navigate to the src folder and run:

python coin_edge.py
python coin_segmentation.py

*Expected Outputs:*  
- Edge_Detection: coints_with_edge_detection.jpeg – Image with detected coin boundaries highlighted and the coin count printed.
 (Edges detected in the coin image)  
- Segmentation: Segmented_coins.jpg – Overall segmented image highlighting coin boundaries.
- segmented_coin.jpg – Individual coin images extracted from the original image.  

### *⿢ Image Stitching (Panorama Creation)*  
Navigate to the src folder and run:

python stitch_panaroma.py


*Expected Outputs:*  
- final_panorama.jpg – Final stitched panorama image.

---

## 📊 Methodology  

### *Part 1: Coin Detection and Segmentation*  
#### 🔹 *Steps Followed:*  
1. Edge Detection (coin_edge.py):
- Convert the image to grayscale and apply Gaussian blur.
- Perform Canny edge detection.
- Use the Hough Circle Transform to detect circular coin shapes.
- Overlay the detected circles on the original image and count the coins. 

2. Segmentation (coin_segmentation.py):

- Convert the image to grayscale and apply Gaussian blur.
-  Apply Otsu’s thresholding (inverted) to create a binary image.
- Use morphological operations to remove noise and enhance the coin regions.
- Compute the distance transform and identify sure foreground and background.
- Apply the Watershed algorithm to segment and isolate individual coins.
- Extract each coin’s region and save it as a separate image.

#### 📌 *Observations:*  
- Edge Detection: Provides robust detection of coin boundaries; however, overlapping coins or noise might cause minor inaccuracies.  
- Segmentation: The Watershed algorithm effectively separates overlapping coins but may require fine-tuning of thresholds and morphological parameters.
---

### *Part 2: Image Stitching*  
#### 🔹 *Steps Followed:*  
1. Load multiple overlapping images.  
2. Convert images to grayscale for feature detection.  
3. Use SIFT to detect keypoints and compute descriptors.  
4. Match keypoints using a FLANN-based matcher and apply Lowe’s ratio test for filtering.  
5. Compute the homography using RANSAC.
6. Warp and stitch images together to form a panorama.  

#### 📌 *Observations:*  
- *Feature Matching:* SIFT combined with FLANN matching works well when images have sufficient overlapping regions and distinct features.  
- *Stitching*: The panorama stitching is successful when the overlap is significant. However, inaccuracies in keypoint matching may lead to slight misalignments, suggesting that future improvements such as multi-band blending could enhance the output.
---


## 🎯 Conclusion  
This project successfully demonstrates the use of *Computer Vision* techniques for:  
- *Object detection and segmentation* (Coin detection).  
- *Feature matching and image stitching* (Panorama generation).  

The methods used provide accurate results, though further improvements like *multi-band blending* for seamless stitching can be explored.  

---

## 📝 Author  
*Raghav Khurana*  
IMT2022550  *