##################################
# Pizza data plotter             #
##################################

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

def plot_rgb_histogram(img, exclude_black=True):
    color = ('b', 'g', 'r')
    plt.figure()
    plt.title('RGB Color Histogram (Excluding Black)' if exclude_black else 'RGB Color Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    
    # Create mask to exclude black pixels
    if exclude_black:
        # Mask where all channels are > 0 (not pure black)
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
    else:
        mask = None
    
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([1, 256])
    plt.show()

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
