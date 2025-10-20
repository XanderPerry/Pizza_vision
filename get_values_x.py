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

def get_LBP(img):
    if img is None:
        print(f"Could not read image by get_LBP")
        return
    
    # P = Number of circularly symmetric neighbor points. (8 for the test only)
    P = 8
    # R = Radius of circle.
    R = 1
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LBP_data = local_binary_pattern(grayed, P, R, method='uniform')

    # Compute histogram (to be able te return the usefull numbers)
    n_bins = int(LBP_data.max() + 1)
    hist, bins = np.histogram(LBP_data.ravel(), bins=n_bins, range=(0, n_bins))

    print("LBP_data:")
    print(LBP_data)
    print("\nhist:")
    print(hist)
    print("\nbins:")
    print(bins)

    return LBP_data, hist, bins