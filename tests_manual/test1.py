# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('balls.png',0)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


