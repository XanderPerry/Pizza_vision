#   This library contains the necessary functions for image pre processing

import numpy as np

import cv2

IMG_W = 640
IMG_H = 480

clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def clahe(img):
    # Convert the image from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the LAB image into separate channels
    lab_planes = list(cv2.split(lab))

    # Apply CLAHE to the L channel
    lab_planes[0] = clahe_obj.apply(lab_planes[0])

    # Merge the channels back
    lab = cv2.merge(tuple(lab_planes))

    # Convert the LAB image back to BGR color space
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img_clahe

def whitebalance(img):
    # Convert the image from BGR to LAB color space
    img_whitebalanced = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Calculate the average values of the 'a' and 'b' channels
    avg_a = np.average(img_whitebalanced[:, :, 1])
    avg_b = np.average(img_whitebalanced[:, :, 2])

    # Adjust the 'a' and 'b' channels to correct color cast
    # The adjustment is proportional to the luminance channel (L)
    luminance_scaled = img_whitebalanced[:, :, 0] / 255.0
    img_whitebalanced[:, :, 1] = img_whitebalanced[:, :, 1] - ((avg_a - 128) * luminance_scaled * 1.1)
    img_whitebalanced[:, :, 2] = img_whitebalanced[:, :, 2] - ((avg_b - 128) * luminance_scaled * 1.1)

    # Clip the 'a' and 'b' channels to [0, 255] to avoid invalid values
    img_whitebalanced[:, :, 1] = np.clip(img_whitebalanced[:, :, 1], 0, 255)
    img_whitebalanced[:, :, 2] = np.clip(img_whitebalanced[:, :, 2], 0, 255)
    img_whitebalanced = img_whitebalanced.astype(np.uint8)
    
    # Convert the image back to BGR color space
    img_whitebalanced = cv2.cvtColor(img_whitebalanced, cv2.COLOR_LAB2BGR)

    return img_whitebalanced

def color_correct(img):
    img_colorcorrected = clahe(img)

    img_colorcorrected = whitebalance(img_colorcorrected)

    return img_colorcorrected

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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if edge_detection == "sobel":
        img_gray = calc_sobel(img_gray)
    elif edge_detection == "laplacian":
        img_gray = calc_laplacian(img_gray)
    elif edge_detection == "canny":
        img_gray = calc_canny(img_gray)
    
    img_circle = hough_detect_circle(img_gray, img)

    return img_circle

def color_mask(img):
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
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

def cut_pizza(img):
    img_cutout = color_correct(img)

    img_cutout = crop_image(img_cutout)

    img_cutout = cutout_circle(img_cutout)

    return img_cutout