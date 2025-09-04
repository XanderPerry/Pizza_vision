# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:34:21 2023

@author: bart.bozon
KLIK HET PLOT WINDOW AAN IN SPYDER!!!!
"""

import cv2 as cv                 #Lib for image processing
import glob
import numpy as np
import matplotlib.pyplot as plt  #Lib for plotting images
import random


def test1():
    # hier stop je je test code die uiteindelijk data oplevert!
    score=np.std(img_array) # voor de grap even de standaarddeviatie
    return random.randint(0,5)#score 

def test2():
    # hier stop je je test code die uiteindelijk data oplevert!
    score=np.average(img_array) # voor de grap even het gemiddelde
    return random.randint(0,5)#score 

#https://www.geeksforgeeks.org/python-get-unique-values-list/
# function to get unique values
def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


img_size = 200
names=[]
labels=[]
test1_outcome=[]
test2_outcome=[]
nummer=[]
i=0
colors=['b','c','g','k,','m','r','w','y']

for name in glob.glob('*.jpg'):
    img_array = cv.imread(name, cv.IMREAD_GRAYSCALE)
    names.append(name)
    label=name.split('_')[0] # we splitten de string en pakken het eerste stukje
    test1_outcome.append(test1())
    test2_outcome.append(test2())
    nummer.append(i)
    i=i+1
    labels.append(label)
    '''
    Dit volgende stukje is niet nodig, maar wel leuk. Snelle manier om foto's te laten zien!
    '''
    img_array_new = cv.resize(img_array, (img_size,img_size))
    plt.imshow(img_array_new, cmap="gray")
    plt.show()
    
# de outcome van de hele test
plt.plot(nummer,test1_outcome,nummer,test2_outcome)
plt.xlabel("# test")
plt.legend(['outcome1','outcome2'] )
plt.show()


name_label_uniq=unique(labels)

for i in range(len(labels)):
    plt.scatter (test1_outcome[i],test2_outcome[i],c=colors[name_label_uniq.index(labels[i])])
    plt.xlabel("outcome 1")
    plt.ylabel("outcome 2")
plt.show()

for i in name_label_uniq:
    print (i,colors[name_label_uniq.index(i)])
    