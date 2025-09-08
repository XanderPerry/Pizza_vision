# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 15:13:31 2025

@author: Xande
"""

import cv2
import numpy as np
import glob

import circle_detector

i = 50
for filename in glob.glob("image_capture/data/sal/train/**/*.jpg", recursive=True):
    if i < 50:
        i+=1 
        continue
    else:
        i=0

    # Load image
    image = cv2.imread(filename)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow("Pizza blurred", blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    cv2.imshow("Pizza hsv", hsv)

    # Define yellow borders
    lower_yellow = np.array([20, 50, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Define red borders part 1
    lower_red1 = np.array([0, 50, 100])
    upper_red1 = np.array([20, 255, 255])
    
    lower_red2 = np.array([160, 50, 100])
    upper_red2 = np.array([179, 255, 255])
    
    # Define red borders part 2
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Combine red and yellow masks
    mask = cv2.bitwise_or(mask_yellow, mask_red)
    
    cv2.imshow("Pizza mask", mask)

    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    
    if contours:
        # Select largest contour
        contour = max(contours, key=cv2.contourArea)
    
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Padding factor
        padding = 0.05  # 5%
    
        # Compute padding in pixels
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Get image hwight and width to avoid going out of bounds
        height, width = image.shape[:2]
    
        # Expand and clamp to image bounds
        x_new = max(x - pad_w, 0)
        y_new = max(y - pad_h, 0)
        x2_new = min(x + w + pad_w, width)
        y2_new = min(y + h + pad_h, height)
    
        # Crop the expanded rectangle
        cropped = image[y_new:y2_new, x_new:x2_new]
        
        # Show the result
        cv2.imshow("Cropped Rectangle", cropped)
        
        # Copy cropped image in grayscale
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge detection algorithms
        cropped_gray_sobel = circle_detector.calc_sobel(cropped_gray)
        cropped_gray_laplacian = circle_detector.calc_laplacian(cropped_gray)
        cropped_gray_canny= circle_detector.calc_canny(cropped_gray)
        
        # Calculate circle detection algorithms
        cropped_hough_circle = circle_detector.hough_detect_circle(cropped_gray, cropped)
        cropped_hough_circle_sobel = circle_detector.hough_detect_circle(cropped_gray_sobel, cropped)
        cropped_hough_circle_laplacian = circle_detector.hough_detect_circle(cropped_gray_laplacian, cropped)
        cropped_hough_circle_canny = circle_detector.hough_detect_circle(cropped_gray_canny, cropped)
        
        # Display results
        cv2.imshow("Cropped", cropped)
        cv2.imshow("Sobel Edge Detection", cropped_gray_sobel)
        cv2.imshow("Laplacian Edge Detection", cropped_gray_laplacian)
        cv2.imshow("Canny Edge Detection", cropped_gray_canny)
        
        cv2.imshow("Hough Detection - Original", cropped_hough_circle)
        cv2.imshow("Hough Detection - Sobel", cropped_hough_circle_sobel)
        cv2.imshow("Hough Detection - Laplacian", cropped_hough_circle_laplacian)
            
    
    # Show result
    cv2.imshow("Pizza Detection", image)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print("Manual interruption by user.")
        break
    
    cv2.destroyAllWindows()
    
cv2.destroyAllWindows()
