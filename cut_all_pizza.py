# This script is used to preprocess the dataset for quicker testing

#   import libraries
import cv2
import glob
import os

import pizza_cutter

#   Define list of represented pizza kinds
KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
#   Define directory path
DIRECTORY = "C:/Users/Xande/EE cursusmateriaal/3A/Beeldherkenning/PizzaVisionRepo/Pizza_vision/"

#   Ask user confirmation to prefent accidental script execution
cut = input("Type 'y' to start cutout process: ")
if cut != 'y':
    exit(0)

#   Loop over pizza kinds
for kind in KINDS:
    #   Print kind currently being processed for progess tracking
    print("starting " + kind)

    #   Loop over, testing, training and validation datasets
    for dataset in ["test", "train", "validation"]:
        #   Initialize counter at 0
        i = 0

        #   Loop over all images in folder
        for filename_old in glob.glob("data/" + kind + "/" + dataset + "/**/*.jpg", recursive=True):
            #   Get new filepath by replacing the "data" folder with "data_cutout"
            filename_new ="data_cutout" + filename_old[filename_old.find("data")+4:]
            
            #   Replace "\\" seperators with "/" for propper image saving
            filename_new = filename_new.replace("\\", "/")
            
            #   Read original image
            img = cv2.imread(filename_old)

            #   Preform pre-processing steps on image
            img_cutout = pizza_cutter.cut_pizza(img)

            #   Get fill filepath of file to safe by joining directory and filepath
            loc = os.path.join(DIRECTORY, filename_new)

            #   Write image to desired location, and get feedback on success
            ret = cv2.imwrite(loc, img_cutout)

            #   Print whether image saving is completed succesfully including filename for debugging and progress tracking
            print(ret, os.path.join(DIRECTORY, filename_new))
