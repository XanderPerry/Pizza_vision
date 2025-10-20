##################################
# Pizza data plotter             #
##################################

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2


def plot_hsv_histogram(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = ('r', 'g', 'b')
    plt.figure()
    plt.title('HSV Color Histogram (Excluding Black)' if exclude_black else 'HSV Color Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')

    # Create mask to exclude black pixels
    if exclude_black:
        # Mask where all channels are > 0 (not pure black)
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
    else:
        mask = None

    for i, col in enumerate(color):
        hist = cv2.calcHist([hsv], [i], mask, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([1, 256])
    plt.show()

def get_mean_hue(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        hue_values = hsv[:,:,0][mask > 0]
    else:
        hue_values = hsv[:,:,0].flatten()
    
    if hue_values.size == 0:
        return None  # No valid hue values found

    mean_hue = np.mean(hue_values)
    return mean_hue

def get_hue_distribution(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        hue_values = hsv[:,:,0][mask > 0]
    else:
        hue_values = hsv[:,:,0].flatten()

    hue_dist = np.bincount(hue_values//10, minlength=18)  # 18 bins for hue (0-179)
    return hue_dist
    
    

def get_mean_sat(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask to exclude black pixels
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        sat_values = hsv[:,:,1][mask > 0]
    else:
        sat_values = hsv[:,:,1].flatten()
    
    if sat_values.size == 0:
        return None  # No valid saturation values found

    mean_sat = np.mean(sat_values)
    return mean_sat

def get_mean_val(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask to exclude black pixels
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        val_values = hsv[:,:,2][mask > 0]
    else:
        val_values = hsv[:,:,2].flatten()

    if val_values.size == 0:
        return None  # No valid value values found
    
    mean_val = np.mean(val_values)
    return mean_val

def get_edge_percentage(img, ignore_black=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if ignore_black:
        mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    edges = cv2.Canny(gray, 40, 100)
    # cv2.imshow("Canny Edges", edges)
    edge_pixels = np.sum(edges > 0)
    total_pixels = np.sum(mask > 0) if ignore_black else img.shape[0] * img.shape[1]
    
    if total_pixels == 0:
        return 0.0  # Avoid division by zero
    
    edge_percentage = (edge_pixels / total_pixels) * 100
    return edge_percentage
