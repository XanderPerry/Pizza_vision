import numpy as np
from skimage.feature import local_binary_pattern

import cv2

def get_mean_hue(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        hue_values = hsv[:,:,0][mask > 0]
    else:
        hue_values = hsv[:,:,0].ravel()
    
    if hue_values.size == 0:
        return None  # No valid hue values found

    mean_hue = np.mean(hue_values)
    return mean_hue

def get_mean_sat(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        sat_values = hsv[:,:,1][mask > 0]
    else:
        sat_values = hsv[:,:,1].ravel()
    
    if sat_values.size == 0:
        return None  # No valid hue values found

    mean_sat = np.mean(sat_values)
    return mean_sat

def get_mean_val(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        val_values = hsv[:,:,2][mask > 0]
    else:
        val_values = hsv[:,:,2].ravel()
    
    if val_values.size == 0:
        return None  # No valid hue values found

    mean_val = np.mean(val_values)
    return mean_val

def get_edge_percentage(img, ignore_black=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 100, 150)
    
    if ignore_black:
        mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    
    cv2.imshow("Canny Edges", edges)
    edge_pixels = np.sum(edges > 0)
    total_pixels = np.sum(mask > 0) if ignore_black else img.shape[0] * img.shape[1]
    
    if total_pixels == 0:
        return 0.0  # Avoid division by zero
    
    edge_percentage = (edge_pixels / total_pixels) * 100
    return edge_percentage