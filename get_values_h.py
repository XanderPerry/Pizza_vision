import numpy as np
from skimage.feature import local_binary_pattern

import cv2



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