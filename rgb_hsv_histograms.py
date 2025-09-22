##################################
# RGB and HSV color histograms  #
##################################

import numpy as np
import matplotlib.pyplot as plt
import glob

import cv2

import pizza_cutter
import pizza_plotter
import whitebalance

for filename in glob.glob("data/haw/train/**/*.jpg", recursive=True):
    # Load image
    img = cv2.imread(filename)
    cv2.imshow("Original image", img)

    # Crop image based on color mask (red/yellow)
    img_cropped = pizza_cutter.crop_image(img)
    cv2.imshow("Cropped image", img_cropped)

    # Crop image to circle using Hough Transform
    img_circle = pizza_cutter.cutout_circle(img_cropped, edge_detection="sobel")
    cv2.imshow("Detected circle", img_circle)

    # Plot RGB and HSV histograms
    pizza_plotter.plot_rgb_histogram(img_circle) 
    pizza_plotter.plot_hsv_histogram(img_circle)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print("Manual interruption by user.")
        break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()  