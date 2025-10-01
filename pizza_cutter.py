import numpy as np
import time

import cv2

import whitebalance

IMG_W = 640
IMG_H = 480


def calc_sobel(img_gray):
    # Apply Sobel operator
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
     
    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    return gradient_magnitude

def calc_laplacian(img_gray):
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
     
    # Convert to uint8
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    
    return laplacian_abs

def calc_canny(img_gray):
    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
     
    # Apply Canny Edge Detector
    edges = cv2.Canny(blur, threshold1=50, threshold2=100)

    return edges

def hough_detect_circle(img_gray, img):
    # Use a faster blur (box filter) instead of medianBlur
    img_gray = cv2.blur(img_gray, (5, 5))

    # Detect circles
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=200,
        param1=100,
        param2=20,
        minRadius=100,
        maxRadius=300
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]

        # Use a single-channel mask for speed
        mask_circle = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask_circle, (x, y), int(r * 1.01), 255, -1)
        output = cv2.bitwise_and(img, img, mask=mask_circle)
        return output

    return img

def cutout_circle(img, edge_detection="sobel"):
    # time_start = time.time()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # time_gray = time.time()

    if edge_detection == "sobel":
        img_gray = calc_sobel(img_gray)
    elif edge_detection == "laplacian":
        img_gray = calc_laplacian(img_gray)
    elif edge_detection == "canny":
        img_gray = calc_canny(img_gray)
    
    # time_edge = time.time()
    
    img_circle = hough_detect_circle(img_gray, img)
    
    # time_end = time.time()
    # print("Duration of gray conversion: " + str(time_gray - time_start))
    # print("Duration of " + edge_detection + " edge detection: " + str(time_edge - time_gray))
    # print("Duration of hough circles: " + str(time_end - time_edge))
    # print("Total duration of circle cropping: " + str(time_end - time_start) + "\n")

    return img_circle

def color_mask(img):
    img_whitebalanced = whitebalance.white_balance_loops(img)

    img_blurred = cv2.GaussianBlur(img_whitebalanced, (5, 5), 0)
    # cv2.imshow("Pizza blurred", img_blurred)
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    # cv2.imshow("Pizza hsv", hsv)

    # Define yellow borders
    upper_yellow = np.array([55, 255, 255])
    lower_yellow = np.array([15, 40, 165])

    # Define red borders part 1
    upper_red1 = np.array([15, 255, 255])
    lower_red1 = np.array([0, 100, 65])

    # Define red borders part 2
    upper_red2 = np.array([180, 255, 255])
    lower_red2 = np.array([160, 40, 60])

    # Combine masks
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Combine red and yellow masks
    result = cv2.bitwise_or(mask_yellow, mask_red)

    return result

def crop_image(img):
    mask = color_mask(img)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Return original image if resulting crop is too small
    if (w < (IMG_W*0.3) or h < (IMG_H*0.4)):
        return img

    # Crop the image using the bounding box
    img_cropped = img[y:y+h, x:x+w]

    return img_cropped

