#   This library provides functions to retrieve a random image from each pizza dataset and 
#       quickly apply functions and image modifications.

import random
import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]

# Store the file lists and current indices globally
file_lists = {}
file_indices = {}
def get_next_images():
    # Initialize file lists and indices on first call
    if not hasattr(get_next_images, "file_lists"):
        get_next_images.file_lists = {}
        get_next_images.file_indices = {}
        for kind in KINDS:

            # Hayan's path #########################################################
            directory = "C:/HU/Jaar3/A/Beeldherkening/data_cutout/" + kind + "/train/"
            # Hayan's path #########################################################
            # directory = "data_cutout/" + kind + "/train/"

            files = sorted(os.listdir(directory))  # sorted for fixed order
            get_next_images.file_lists[kind] = files
            get_next_images.file_indices[kind] = 0

    imgs = {}
    filenames = {}
    for kind in KINDS:
        files = get_next_images.file_lists[kind]
        index = get_next_images.file_indices[kind]

        filename = os.path.join(f"C:/HU/Jaar3/A/Beeldherkening/data_cutout/{kind}/train/", files[index])
        img = cv2.imread(filename)

        imgs[kind] = img
        filenames[kind] = filename

        # Move index to next file, wrap around at end
        get_next_images.file_indices[kind] = (index + 1) % len(files)

    return imgs


def get_random_images(show_names=True, dataset="data_cutout", datagroup="train"):
    imgs = {}
    for kind in KINDS:
        # # Hayan's path #########################################################
        # directory = "C:/HU/Jaar3/A/Beeldherkening/data set/data/" + kind + "/" + datagroup + "/"
        # # Hayan's path #########################################################
        directory = dataset + "/" + kind + "/" + datagroup + "/"
        filename = directory + random.choice(os.listdir(directory))
        imgs[kind] = cv2.imread(filename)

        if show_names:
            print(kind + ": " + filename)
            if kind == KINDS[-1]:
                print()
    
    return imgs

def modify_imgs(func, imgs):
    imgs_modified = {}

    for kind in KINDS:
        imgs_modified[kind] = func(imgs[kind])

    return imgs_modified

def apply_function(func, imgs):
    imgs_results = {}

    for kind in KINDS:
        result = func(imgs[kind])
        imgs_results[kind] = result

    return imgs_results

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

def imgs_print_results(results_list, labels_list):
    for kind in KINDS:
        print(kind + ": ")
        for i in range(len(results_list)):
            print("\t" + labels_list[i] + " = " + str(results_list[i][kind]))

    print()
    return
