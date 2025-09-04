# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Count number of white pixels in mask array
    n_white = 0
    for row in mask:
        for pixel in row:
            if pixel == 255:
                n_white += 1
        
    # Display blue detection results
    if n_white > 2000:
        print("Blue object detected!")
        
        frame[0:20, 0:20] = (250,50,50)
        
    print(f"Mask contains {n_white} white pixels.\n")
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
