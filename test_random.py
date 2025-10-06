import random
import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]

def get_random_images(n_images = 1):
    imgs = {}

    for kind in KINDS:
        directory = "data_cutout/" + kind + "/train/"
        filename = directory + random.choice(os.listdir(directory))
        imgs[kind] = cv2.imread(filename)

    return imgs

def apply_function(func, imgs):
    imgs_modified = {}

    for kind in KINDS:
        imgs_modified[kind] = func(imgs[kind])

    return imgs_modified

def imgs_compare_visual(imgs_a, imgs_b, label_a = " before", label_b = " after"):
    plt.figure(figsize=(10, 20))

    for i, kind in enumerate(KINDS):
        plt.subplot(3, 4, 2*i+1)
        plt.imshow(cv2.cvtColor(imgs_a[kind], cv2.COLOR_BGR2RGB))
        plt.title(kind + label_a, fontsize=8)
        plt.axis('off')

        plt.subplot(3, 4, 2*i+2)
        plt.imshow(cv2.cvtColor(imgs_b[kind], cv2.COLOR_BGR2RGB))
        plt.title(kind + label_b, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)

def imgs_compare_visual_list(imgs_list, labels_list):
    n = len(imgs_list)

    plt.figure(figsize=(15, 2*n))

    for i in range(n):
        for j, kind in enumerate(KINDS):
            plt.subplot(n, 6, (i*6+j)+1)
            plt.imshow(cv2.cvtColor(imgs_list[i][kind], cv2.COLOR_BGR2RGB))
            plt.title(kind + labels_list[i], fontsize=8)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)
