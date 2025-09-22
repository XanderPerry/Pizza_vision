# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 15:13:31 2025

@author: Xande
"""

import glob
import cv2
import numpy as np
#import copy

i = 50
for filename in glob.glob("data/che/train/**/*.jpg", recursive=True):
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
    
    # Draw contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50000:  # Ignore small areas
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
    
    # Show result
    cv2.imshow("Pizza Detection", image)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print("Manual interruption by user.")
        break
    
    cv2.destroyAllWindows()
    
cv2.destroyAllWindows()
