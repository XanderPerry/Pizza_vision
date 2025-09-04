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

#snij een selectie uit het plaatje
# dit maakt een copy van het adres van het plaatje:
selection1 = image[10:150, 10:200]

# dit maakt een echte copy van een deel van het plaatje:
selection2 = copy.deepcopy(image[10:150, 10:200])

# de kleuren kun je zetten door een zogenaamd tupplet van kleur (50,250,50)
image[10:150, 10:200] = (50,250,50)

# Toon het beeld in een venster
cv2.imshow("Cut-out1", selection1)
cv2.imshow("Cut-out2", selection2)
cv2.imshow("Input image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()