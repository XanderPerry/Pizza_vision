# Generated using CoPilot
#
# This script is used to quickly visually test different colormasks

import cv2
import numpy as np
import random
import os

import pizza_cutter

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
i = 0

# Load the image
directory = "data/" + KINDS[i] + "/train/"
i=(i+1)%6
filename = directory + random.choice(os.listdir(directory))
img = cv2.imread(filename)

img_clahe = pizza_cutter.clahe(img)

img_whitebalanced = pizza_cutter.whitebalance(img_clahe)

hsv = cv2.cvtColor(img_whitebalanced, cv2.COLOR_BGR2HSV)

def nothing(x):
    pass

# Create separate windows for red and yellow controls
cv2.namedWindow('Red Mask 1 Controls', cv2.WINDOW_NORMAL)
cv2.namedWindow('Red Mask 2 Controls', cv2.WINDOW_NORMAL)
cv2.namedWindow('Yellow Mask Controls', cv2.WINDOW_NORMAL)

# Red mask 1 sliders
cv2.createTrackbar('Enable Mask', 'Red Mask 1 Controls', 1, 1, nothing)
cv2.createTrackbar('Lower H', 'Red Mask 1 Controls', 0, 180, nothing)
cv2.createTrackbar('Lower S', 'Red Mask 1 Controls', 100, 255, nothing)
cv2.createTrackbar('Lower V', 'Red Mask 1 Controls', 65, 255, nothing)
cv2.createTrackbar('Upper H', 'Red Mask 1 Controls', 15, 180, nothing)
cv2.createTrackbar('Upper S', 'Red Mask 1 Controls', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Red Mask 1 Controls', 255, 255, nothing)

# Red mask 2 sliders
cv2.createTrackbar('Enable Mask', 'Red Mask 2 Controls', 1, 1, nothing)
cv2.createTrackbar('Lower H', 'Red Mask 2 Controls', 160, 180, nothing)
cv2.createTrackbar('Lower S', 'Red Mask 2 Controls', 40, 255, nothing)
cv2.createTrackbar('Lower V', 'Red Mask 2 Controls', 60, 255, nothing)
cv2.createTrackbar('Upper H', 'Red Mask 2 Controls', 180, 180, nothing)
cv2.createTrackbar('Upper S', 'Red Mask 2 Controls', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Red Mask 2 Controls', 255, 255, nothing)

# Yellow mask sliders
cv2.createTrackbar('Enable Mask', 'Yellow Mask Controls', 1, 1, nothing)
cv2.createTrackbar('Lower H', 'Yellow Mask Controls', 15, 180, nothing)
cv2.createTrackbar('Lower S', 'Yellow Mask Controls', 40, 255, nothing)
cv2.createTrackbar('Lower V', 'Yellow Mask Controls', 165, 255, nothing)
cv2.createTrackbar('Upper H', 'Yellow Mask Controls', 55, 180, nothing)
cv2.createTrackbar('Upper S', 'Yellow Mask Controls', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Yellow Mask Controls', 255, 255, nothing)

while True:
    red1_enabled = cv2.getTrackbarPos('Enable Mask', 'Red Mask 1 Controls') == 1
    red2_enabled = cv2.getTrackbarPos('Enable Mask', 'Red Mask 2 Controls') == 1
    yellow_enabled = cv2.getTrackbarPos('Enable Mask', 'Yellow Mask Controls') == 1

    # Red mask 1 values
    r1_lower = np.array([cv2.getTrackbarPos('Lower H', 'Red Mask 1 Controls'),
                        cv2.getTrackbarPos('Lower S', 'Red Mask 1 Controls'),
                        cv2.getTrackbarPos('Lower V', 'Red Mask 1 Controls')])
    r1_upper = np.array([cv2.getTrackbarPos('Upper H', 'Red Mask 1 Controls'),
                        cv2.getTrackbarPos('Upper S', 'Red Mask 1 Controls'),
                        cv2.getTrackbarPos('Upper V', 'Red Mask 1 Controls')])
    
    r2_lower = np.array([cv2.getTrackbarPos('Lower H', 'Red Mask 2 Controls'),
                        cv2.getTrackbarPos('Lower S', 'Red Mask 2 Controls'),
                        cv2.getTrackbarPos('Lower V', 'Red Mask 2 Controls')])
    r2_upper = np.array([cv2.getTrackbarPos('Upper H', 'Red Mask 2 Controls'),
                        cv2.getTrackbarPos('Upper S', 'Red Mask 2 Controls'),
                        cv2.getTrackbarPos('Upper V', 'Red Mask 2 Controls')])

    # Yellow mask values
    y_lower = np.array([cv2.getTrackbarPos('Lower H', 'Yellow Mask Controls'),
                        cv2.getTrackbarPos('Lower S', 'Yellow Mask Controls'),
                        cv2.getTrackbarPos('Lower V', 'Yellow Mask Controls')])
    y_upper = np.array([cv2.getTrackbarPos('Upper H', 'Yellow Mask Controls'),
                        cv2.getTrackbarPos('Upper S', 'Yellow Mask Controls'),
                        cv2.getTrackbarPos('Upper V', 'Yellow Mask Controls')])

    # Create masks
    red_mask1 = cv2.inRange(hsv, r1_lower, r1_upper)
    red_mask2 = cv2.inRange(hsv, r2_lower, r2_upper)
    yellow_mask = cv2.inRange(hsv, y_lower, y_upper)

    # Create color overlays
    red1_overlay = np.zeros_like(img)
    red1_overlay[:] = (0, 0, 255)  # Bright red in BGR
    red2_overlay = np.zeros_like(img)
    red2_overlay[:] = (28, 153, 255)  # Bright orange in BGR    
    yellow_overlay = np.zeros_like(img)
    yellow_overlay[:] = (0, 255, 255)  # Bright yellow in BGR

    # Apply masks to overlays
    red1_highlight = cv2.bitwise_and(red1_overlay, red1_overlay, mask=red_mask1)
    red2_highlight = cv2.bitwise_and(red2_overlay, red2_overlay, mask=red_mask2)
    yellow_highlight = cv2.bitwise_and(yellow_overlay, yellow_overlay, mask=yellow_mask)

    # Combine overlays with original image
    combined = img_whitebalanced.copy()
    if red1_enabled: 
        combined = cv2.addWeighted(combined, 1.0, red1_highlight, 1, 0)
    if red2_enabled: 
        combined = cv2.addWeighted(combined, 1.0, red2_highlight, 1, 0)
    if yellow_enabled: 
        combined = cv2.addWeighted(combined, 1.0, yellow_highlight, 1, 0)

    # Show result
    cv2.imshow('Original', img)
    cv2.imshow('Whitebalanced', img_whitebalanced)
    cv2.imshow('Overlay Result', combined)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('n'):
        print("New image.")
        directory = "data/" + KINDS[i] + "/train/"
        i=(i+1)%6
        filename = directory + random.choice(os.listdir(directory))
        img = cv2.imread(filename)
        img_clahe = pizza_cutter.clahe(img)

        img_whitebalanced = pizza_cutter.whitebalance(img_clahe)

        hsv = cv2.cvtColor(img_whitebalanced, cv2.COLOR_BGR2HSV)

    elif key & 0xFF == ord('s'):
        print("Used mask parameters:")
        print("Red1 upper:  " + str(r1_upper))
        print("Red1 lower:  " + str(r1_lower))
        print("Red2 upper:  " + str(r2_upper))
        print("Red2 lower:  " + str(r2_lower))
        print("Yellow upper:" + str(y_upper))
        print("Yellow lower:" + str(y_lower))
        
    elif key & 0xFF == 27:
        cv2.destroyAllWindows()
        print("Exiting...")
        break

cv2.destroyAllWindows()

