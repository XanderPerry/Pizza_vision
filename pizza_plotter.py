##################################
# Pizza data plotter             #
##################################

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import cv2

import add_to_dataframe

sns.set_theme(style="whitegrid")

def get_df():
    opened = False

    while not opened:
        try:
            filename = add_to_dataframe.DIRECTORY+input("What df would you like to use? ")
            global pizza_df
            pizza_df = pd.read_csv(filename, delimiter=',')
        except:
            print("File not found, try again.")
            continue
        opened = True

    print("File opened")
    print(pizza_df.head())

def get_axes():
    print("choose from: " + str(list(pizza_df)))
    x = input("What value should the x axis show? ")
    while not (x in list(pizza_df)):
        print("Invalid x value, choose from: " + str(list(pizza_df)))
        x = input("What value should the x axis show?")

    y = input("What value should the y axis show? ")
    while not (y in list(pizza_df)):
        print("Invalid y value, choose from: " + str(list(pizza_df)))
        y = input("What value should the x axis show? ")

    return x, y

def get_ax():
    print("choose from: " + str(list(pizza_df)))
    ax = input("What value should the plot show? ")
    while not (ax in list(pizza_df)):
        print("Invalid value, choose from: " + str(list(pizza_df)))
        ax = input("What value should the plot show? ")
    return ax

def scatterplot():
    x_column, y_column = get_axes()

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.scatterplot(x=pizza_df[x_column], y=pizza_df[y_column],
                    hue=pizza_df["kind"])

    plt.suptitle(x_column+" vs "+y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    sns.despine(f, left=True, bottom=True)
    plt.show()

def violinplot():
    y = get_ax()
    
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    
    sns.violinplot(x="kind", y=y, hue="kind", data=pizza_df)

    sns.despine(f, left=True, bottom=True)
    plt.show()

if __name__ == "__main__":
    get_df()

    new_plot = True

    while new_plot:
        plotkind = input("What plot would you like? ").lower()

        if plotkind == "scatter":
            scatterplot()
        if plotkind == "violin":
            violinplot()
        else:
            print("Plotkind unknown...")

        if input("would you like another plot?(y/n) ").lower() in ["y", "yes"]:
            new_plot = True
        else:
            new_plot = False