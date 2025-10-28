#   This library provides functions to retrieve a random image from each pizza dataset and 
#       quickly apply functions and image modifications.

#   Library imports
import random
import os
import matplotlib.pyplot as plt

import cv2

#   Define list of pizza kinds
KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]

# Store the file lists and current indices globally
file_lists = {}
file_indices = {}
def get_next_images():
    """This function returns a dict of the next image per pizza kind from the test dataset"""
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
    """This function returns a dict of one random image per pizza kind"""
    #   Initialize empty dict
    imgs = {}
    #   Loop over pizza kinds
    for kind in KINDS:
        #   Get Filename of random image
        directory = dataset + "/" + kind + "/" + datagroup + "/"
        filename = directory + random.choice(os.listdir(directory))

        #   Write image to dictionary
        imgs[kind] = cv2.imread(filename)

        #   Print filenames for traceability
        if show_names:
            print(kind + ": " + filename)
    
    #   Return images dictionary
    return imgs

def modify_imgs(func, imgs):
    """This function modifies images in the given dictionary with the given function and returns a dictionary of the modified images"""
    #   Initialize empty dict
    imgs_modified = {}

    #   Loop over pizza kinds
    for kind in KINDS:
        #   Modify image and add to dictionary
        imgs_modified[kind] = func(imgs[kind])

    #   Return modified dictionary
    return imgs_modified

def apply_function(func, imgs):
    """This function applies a given function to images in the given dictionary and returns a dictionary of the results"""
    #   Initialize empty dict 
    imgs_results = {}

    #   Loop over pizza kinds
    for kind in KINDS:
        #   Apply function and add results to results dicitionary
        result = func(imgs[kind])
        imgs_results[kind] = result

    #   Return results dicitionary
    return imgs_results

def imgs_compare_visual(imgs_a, imgs_b, label_a = " before", label_b = " after"):
    """This function shows images from two dicts side by side for visual comparison"""
    #   Initialize figure
    plt.figure(figsize=(10, 20))

    #   Loop over enumerated pizza kinds
    for i, kind in enumerate(KINDS):
        #   Place images in figure grid with corresponding images of different dicts side by side
        plt.subplot(3, 4, 2*i+1)
        plt.imshow(cv2.cvtColor(imgs_a[kind], cv2.COLOR_BGR2RGB))
        plt.title(kind + label_a, fontsize=8)
        plt.axis('off')

        plt.subplot(3, 4, 2*i+2)
        plt.imshow(cv2.cvtColor(imgs_b[kind], cv2.COLOR_BGR2RGB))
        plt.title(kind + label_b, fontsize=8)
        plt.axis('off')
    
    #   Adjust layout and show images
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)

def imgs_compare_visual_list(imgs_list, labels_list):
    """This function shows images from multiple dicts side by side for visual comparison"""
    #   Get amount of lists to compare
    n = len(imgs_list)

    #   Initialize figure (adjust size based on amount of images)
    plt.figure(figsize=(15, 2*n))

    #   Plase images in figure grid with corresponding images of different dicts side by side
    for i in range(n):
        for j, kind in enumerate(KINDS):
            plt.subplot(n, 6, (i*6+j)+1)
            plt.imshow(cv2.cvtColor(imgs_list[i][kind], cv2.COLOR_BGR2RGB))
            plt.title(kind + labels_list[i], fontsize=8)
            plt.axis('off')
    
    #   Adjust layout and show images
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)

def imgs_print_results(results_list, labels_list):
    """This function prints the results from one or more lists"""
    #   Loop over pizza kinds
    for kind in KINDS:
        #   Print pizza kind followed by coressponding results and labels
        print(kind + ": ")
        for i in range(len(results_list)):
            print("\t" + labels_list[i] + " = " + str(results_list[i][kind]))

    #   Print newline
    print()
