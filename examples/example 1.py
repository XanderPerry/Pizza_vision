# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:32:27 2020

@author: bart.bozon
"""

import cv2
import glob

for filename in glob.glob('*.jpg'):
    bigimage = cv2.imread(filename,0)
    scale_percent = 100 # percent of original size
    width = int(bigimage.shape[1] * scale_percent / 100)
    height = int(bigimage.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv2.resize(bigimage, dim, interpolation = cv2.INTER_AREA)
    orig_image=image.copy()      
    cv2.imshow("first", image)
    filtered_image = image.copy()
    radius=5 
    filtered_image = cv2.GaussianBlur(filtered_image, (radius,radius), 0)
    cv2.imshow("filtered", filtered_image)
    for i in range(20):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(filtered_image)
        print (maxLoc,maxVal)
        cv2.circle(filtered_image, maxLoc, radius*7, (0, 0, 0), -1)
        cv2.circle(image, maxLoc, 10, (255, 0, 0), 1)
    print("next")
    cv2.imshow("Output", image)
    cv2.imshow("Final_filtered", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
print("finished")