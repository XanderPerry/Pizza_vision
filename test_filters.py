#   Generated with help from CoPilot

#   This script unleashes a fuckton of filters on a image in hopes of finding something usefull

import random
import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]

# Helper functions
def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    return (c * np.log(1 + img)).astype(np.uint8)

def unsharp_mask(img):
    blur = cv2.GaussianBlur(img, (9, 9), 10)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

def saturation_boost(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 50)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Define filters Written by CoPilot
def apply_filters(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    filters = {
        'Original': img_rgb,
        'Grayscale': gray,
        'Histogram Equalization': cv2.equalizeHist(gray),
        'CLAHE': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray),
        'Gamma Correction': gamma_correction(img_rgb),
        'Log Transform': log_transform(gray),
        'Normalization': cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX),
        'Thresholding': cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
        'Adaptive Threshold': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2),
        'Bitwise NOT': cv2.bitwise_not(gray),
        'Gaussian Blur': cv2.GaussianBlur(img_rgb, (5, 5), 0),
        'Median Blur': cv2.medianBlur(img_rgb, 5),
        'Bilateral Filter': cv2.bilateralFilter(img_rgb, 9, 75, 75),
        'Morph Open': cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((5,5), np.uint8)),
        'Morph Close': cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8)),
        'Sobel X': cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
        'Sobel Y': cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3),
        'Scharr': cv2.Scharr(gray, cv2.CV_64F, 1, 0),
        'Laplacian': cv2.Laplacian(gray, cv2.CV_64F),
        'Canny': cv2.Canny(gray, 100, 200),
        'Unsharp Mask': unsharp_mask(img_rgb),
        'High-pass Filter': high_pass_filter(img_rgb),
        'Channel Split (Red)': cv2.split(img_rgb)[0],
        'Channel Split (Green)': cv2.split(img_rgb)[1],
        'Channel Split (Blue)': cv2.split(img_rgb)[2],
        'HSV Conversion': cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV),
        'LAB Conversion': cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB),
        'White Balance': white_balance(img_rgb),
        'Saturation Boost': saturation_boost(img_rgb),
        'Bitwise AND with mask': cv2.bitwise_and(img_rgb, img_rgb, mask=cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]),
    }
    return filters

def display_filters(filters, cols=8, figsize=(15, 20)):
    rows = -(-len(filters) // cols)  # Ceiling division
    plt.figure(figsize=figsize)
    
    for i, (name, result) in enumerate(filters.items()):
        plt.subplot(rows, cols, i + 1)
        if len(result.shape) == 2:  # Grayscale
            plt.imshow(result, cmap='gray')
        elif result.shape[2] == 3:  # Color
            plt.imshow(result)
        else:  # Unexpected format
            plt.imshow(result, cmap='gray')
        plt.title(name, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

for kind in KINDS:
    directory = "data/" + kind + "/train/"
    filename = directory + random.choice(os.listdir("data/" + kind + "/train"))
    img = cv2.imread(filename)

    # Apply filters
    filters = apply_filters(img)

    # Display results
    display_filters(filters)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print("Manual interruption by user.")
        cv2.destroyAllWindows()
        break
    elif key & 0xFF == 27:
        cv2.destroyAllWindows()
        print("Exiting...")
        exit(0)

cv2.destroyAllWindows()
