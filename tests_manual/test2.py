# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import copy

image = cv2.imread('balls.png')

if image is None:
    print("\nERROR: Could not open image\n")
    exit()
    
# Bepaal afmetingen en aantal kleurkanalen
height = image.shape[0]
width = image.shape[1]
colors = image.shape[2]

print ("%d pixels breed" % width)
print ("%d pixels hoog" % height)
print ("%d kleur kanalen" % colors)

image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#RGB - Blue
cv2.imshow('B-RGB',image_RGB[:, :, 0])
# RGB - Green
cv2.imshow('G-RGB',image_RGB[:, :, 1])
# RGB Red
cv2.imshow('R-RGB',image_RGB[:, :, 2])
# HSV - H
cv2.imshow('H-HSV',image_HSV[:, :, 0])
# HSV - S
cv2.imshow('S-HSV',image_HSV[:, :, 1])
# HSV - V

cv2.imshow('V-HSV',image_HSV[:, :, 2])
cv2.imshow("HSV", image_HSV)
cv2.imshow("RGB", image_RGB)
cv2.imshow("Input image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()