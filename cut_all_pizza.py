# This script is used to preprocess the dataset for quicker testing

import cv2
import glob
import os

import pizza_cutter

DIRECTORY = "C:/Users/Xande/EE cursusmateriaal/3A/Beeldherkenning/PizzaVisionRepo/Pizza_vision/"
KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]


cut = input("Type 'y' to start cutout process: ")
if cut != 'y':
    exit(0)

for kind in KINDS:
    print("starting " + kind)
    for dataset in ["test", "train", "validation"]:
        i = 0
        for filename_old in glob.glob("data/" + kind + "/" + dataset + "/**/*.jpg", recursive=True):
            filename_new ="data_cutout" + filename_old[filename_old.find("data")+4:]
            filename_new = filename_new.replace("\\", "/")
            
            img = cv2.imread(filename_old)

            img_cutout = pizza_cutter.cut_pizza(img)

            loc = os.path.join(DIRECTORY, filename_new)

            ret = cv2.imwrite(loc, img_cutout)

            print(ret, os.path.join(DIRECTORY, filename_new))
