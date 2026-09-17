# Pill & Coin Counter with Color-Based Sorting System

## Overview
This project is an end-to-end Computer Vision pipeline designed to detect, segment, count, and classify overlapping or touching objects (pills and coins) using classical Image Processing and Machine Learning techniques. 

The application removes noise, handles uneven lighting, separates touching boundaries using Watershed Segmentation, extracts visual features, and automatically categorizes items into distinct groups without requiring pre-labeled training data.

---

## Technical Architecture & Methodology

The pipeline processes input images through five key stages:

1. **Preprocessing & Illumination Correction:** 
   - Converts input image to Grayscale.
   - Applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to normalize exposure across unevenly lit areas.
   - Applies **Gaussian Blurring** and **Morphological Opening** (`cv2.MORPH_OPEN`) to filter out high-frequency background noise.

2. **Segmentation & Object Isolation:**
   - Computes an **Euclidean Distance Transform** (`cv2.distanceTransform`) on the thresholded foreground.
   - Applies **Watershed Algorithm** (`cv2.watershed`) using connected components as markers to split touching objects cleanly.

3. **Feature Extraction:**
   - Extracts mean **HSV (Hue, Saturation, Value)** color attributes for each isolated contour.
   - Calculates geometric **circularity metrics** ($4\pi \times \text{Area} / \text{Perimeter}^2$) and contour area.

4. **Dimensionality Reduction & Clustering:**
   - Standardizes extracted feature vectors and performs **PCA (Principal Component Analysis)** to project features down to 2 principal components.
   - Clusters reduced features using **K-Means Clustering** to segregate objects by category.
   - Trains a **K-Nearest Neighbors (KNN)** model for item label predictions.

5. **Visualization & Metrics HUD:**
   - Draws green contour boundaries around detected objects.
   - Displays real-time on-screen Head-Up Display (HUD) showing total object counts and item counts grouped by cluster category.
   - Saves a side-by-side comparison image: `[ Original Frame | Binary Mask | Processed Output ]`.

---

## Project Structure

```text
pill-coin-counter/
├── main.py              # Core executable script containing code & synthetic test data generator
├── README.md            # Comprehensive project setup and execution documentation
└── pill_coin_counter_output.png  # Rendered output image (generated upon execution)
