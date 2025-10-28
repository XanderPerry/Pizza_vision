#   This script is used to automatically plot values from csv files

#   Library imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import add_to_dataframe

#   Set seaborn theme
sns.set_theme(style="whitegrid")

def get_df():
    """This function opens a dataframe from a .csv file"""
    #   Ask for user input until accepted name is found
    opened = False
    while not opened:
        #   Catch errors from non-openable files
        try:
            filename = add_to_dataframe.DIRECTORY+input("What df would you like to use? ")
            global pizza_df
            pizza_df = pd.read_csv(filename, delimiter=',')
        except:
            print("File not found, try again.")
            continue
        opened = True

    #   Print first lines of opened dataframe for testing
    print("File opened")
    print(pizza_df.head())

def get_axes():
    """This function returns feature for the x and y axes from available features"""
    #   Ask for x-axes feature
    print("choose from: " + str(list(pizza_df)))
    x = input("What value should the x axis show? ")

    #   Request new input if invalid feature is chosen
    while not (x in list(pizza_df)):
        print("Invalid x value, choose from: " + str(list(pizza_df)))
        x = input("What value should the x axis show?")

    #   Ask for y-axes feature
    y = input("What value should the y axis show? ")
    #   Request new input if invalid feature is chosen
    while not (y in list(pizza_df)):
        print("Invalid y value, choose from: " + str(list(pizza_df)))
        y = input("What value should the x axis show? ")

    #   Return desired features
    return x, y

def get_ax():
    """This function returns feature from available features"""
    #   Ask for feature
    print("choose from: " + str(list(pizza_df)))
    ax = input("What value should the plot show? ")

    #   Request new input if invalid feature is chosen
    while not (ax in list(pizza_df)):
        print("Invalid value, choose from: " + str(list(pizza_df)))
        ax = input("What value should the plot show? ")
    return ax

def scatterplot():
    """This function creates a scatterplot of two desired features"""
    #   Get desired features
    x_column, y_column = get_axes()

    #   Make scatterplot of desired functions
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.scatterplot(x=pizza_df[x_column], y=pizza_df[y_column],
                    hue=pizza_df["kind"])

    #   Add plot layout
    plt.suptitle(x_column+" vs "+y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    sns.despine(f, left=True, bottom=True)
    
    #   Show plot
    plt.show()

def violinplot():
    """This function creates a violinplot of desired feature"""
    #   Get desired feature
    y = get_ax()
    
    #   Make violinplot of desired feature
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.violinplot(x="kind", y=y, hue="kind", data=pizza_df)

    #   Add plot layout
    plt.suptitle(y+" across kinds")
    sns.despine(f, left=True, bottom=True)

    #   Show plot
    plt.show()

#   Only run if file is __main__
if __name__ == "__main__":
    #   Get source dataframe
    get_df()

    #   Generate new plot while user requests
    new_plot = True
    while new_plot:
        #   Ask for desired plot
        plotkind = input("What plot would you like? ").lower()

        #   Generate desired plot, skip if plotkind unknown
        if plotkind == "scatter":
            scatterplot()
        if plotkind == "violin":
            violinplot()
        else:
            print("Plotkind unknown...")

        #   Request if user desires another plot
        if input("would you like another plot?(y/n) ").lower() in ["y", "yes"]:
            new_plot = True
        else:
            new_plot = False