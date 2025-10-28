# This script is used quickly add feature data to a .csv file.
#   The function arguments in the __main__ should be replaced with the desired
#   feature extraction function and the label of the feature. The user then gets
#   prompted to select a .csv to add the data to and to choose a name for the
#   resulting .csv.

#   Library imports
import pandas as pd
import glob
import os
import cv2

import get_values_x
import get_values_h

#   Define list of represented pizza kinds
KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
#   Define directory of dataframes folder
DIRECTORY = "pizza_dataframes/"
    
def filename_to_id(filename):
    """Returns string of ID corresponding to given filename"""
    """ID consists of {kind abbreviation}+{number}, eg. the 99th salami pizza is: sal0099"""
    ID = filename[filename.find("/")+1:filename.find("/")+4] + filename[-8: -4]

    return ID

def create_initial():
    """Creates the initial .csv file for feature data, this file only contains the collumns for image ID and the true kind"""
    #   Create pandas dataframe
    pizza_df = pd.DataFrame(columns=["ID", "kind"])

    #   Loop over the pizza kinds
    for kind in KINDS:
        #   Print the start of kind processing for progress tracking
        print(f"Processing {kind} pizzas...")

        #   Loop over all pizzas within Pizza kind training data
        for filename in glob.glob("data_cutout/" + kind + "/train/**/*.jpg", recursive=True):
            #   Add row to the dataframe with the corresponding image ID and true kind
            pizza_df.loc[len(pizza_df)] = [
                filename_to_id(filename),
                kind
                ]

    #   Save the created dataframe under the name "Pizza.csv"
    pizza_df.to_csv(DIRECTORY + 'Pizza.csv', index=False) 

def add_values(func, label):
    """Add features extracted by input function to a feature dataframe under the given label"""
    #   Get user input for source .csv file
    name_old = DIRECTORY+input("What df would you like to add data to? ")
    
    #   Error handling for non-existing files
    try:  
        df_old = pd.read_csv(name_old)
    except:
        print("Old df not found")
        exit(0)

    #   Get user input for destination .csv file
    name_new = DIRECTORY+input("What should the new df be called? ")

    #   Check whether file exists to prevent overwriting files
    if os.path.exists(name_new):
        print("Destination file already exists.")
        exit(0)
    #   Check whethet the correct file extention is used
    elif not name_new[-4:] == ".csv":
        print(name_new[-4:])
        print("File extention should be .csv")
        exit(0)

    #   Make copy of existing df to add data to
    df_new = df_old.copy()

    #   Create dataframe with the data which should be added
    df_to_add = pd.DataFrame(columns=["ID", label])

    #   Loop over the pizza kinds
    for kind in KINDS:
        #   Loop over all pizzas within Pizza kind training data
        for filename in glob.glob("data_cutout/" + kind + "/train/**/*.jpg", recursive=True):
            #   Print the filename of the image to process for debugging and progress tracking
            print(f"Processing: {filename}")
            
            #   Read image
            img = cv2.imread(filename)

            #   Get image ID
            id = filename_to_id(filename)
            #   Get output of given function
            new_value = func(img)

            #   Add row to dataframe with corresponding image ID and feature value
            df_to_add.loc[len(df_to_add)] = [
                id,
                new_value
                ]

    #   Merge the copied dataframe and the dataframe with new values based on image ID
    df_new = pd.merge(df_new, df_to_add, on="ID", how="left")

    #   Save resulting dataframe under the name chosen by the user
    df_new.to_csv(name_new, index=False)

    #   Print first rows of resulting dataframe to check for successfull addition
    print(df_new.head())
 
#   Only execute if file is main (not when loading library)
if __name__ == "__main__":    
    add_values(get_values_x.get_pizza_radius, "pizza_radius")
