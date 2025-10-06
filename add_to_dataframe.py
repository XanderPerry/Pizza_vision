##################################
# RGB and HSV color histograms  #
##################################

import numpy as np
import pandas as pd
import glob
import os
import cv2

import get_values_x

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
DIRECTORY = "pizza_dataframes/"

def filename_to_id(filename):
    return filename[filename.find("/")+1:filename.find("/")+4] + filename[-8: -4]

def create_initial():
    pizza_df = pd.DataFrame(columns=["ID", "kind"])

    for kind in KINDS:
        print(f"Processing {kind} pizzas...")

        for filename in glob.glob("data_cutout/" + kind + "/train/**/*.jpg", recursive=True):
            pizza_df.loc[len(pizza_df)] = [
                filename_to_id(filename),
                kind
                ]

    pizza_df.to_csv(DIRECTORY + 'Pizza.csv', index=False) 

def add_values(func, label):
    name_old = DIRECTORY+input("What df would you like to add data to? ")
    try:  
        df_old = pd.read_csv(name_old)
    except:
        print("Old df not found")
        exit(0)

    name_new = DIRECTORY+input("What should the new df be called? ")
    if os.path.exists(name_new):
        print("Destination file already exists.")
        exit(0)
    elif not name_new[-4:] == ".csv":
        print(name_new[-4:])
        print("File extention should be .csv")
        exit(0)

    df_new = df_old.copy()

    df_to_add = pd.DataFrame(columns=["ID", label])
    for kind in KINDS:
        for filename in glob.glob("data_cutout/" + kind + "/train/**/*.jpg", recursive=True):
            img = cv2.imread(filename)
            print(f"Processing: {filename}")
            id = filename_to_id(filename)
            new_value = func(img)

            df_to_add.loc[len(df_to_add)] = [
                id,
                new_value
                ]

    df_new = pd.merge(df_new, df_to_add, on="ID", how="left")

    df_new.to_csv(name_new, index=False)
    print(df_new.head())
    
    return
    

add_values(get_values_x.get_mean_hue, "mean_hue")
