# This library contains the functions for feature extraction written by Hayan, all functions take a image as input and output 
#   one value so they can be used in the functions from test_random.py and add_to_dataframe.py

# Librarie imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Predefined image names for the ploting.
KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
# internal counter to track which name to use next
_kind_index = 0 
# store (name, mean, variance) for scatter plot
_summary_data = []  
def get_red_percentages(img):
    """This function returns the total red percentage of the input image"""
    if img is None:
        return None
    # Range of darker red to get the biggest differance between the sets.
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])


    # Convert from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Exclude very dark/black pixels which helps exclude the background that would be already made black in another function.
    exclude_black_pixels = hsv[:, :, 2] > 20  

    # Create masks for the red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Calculate the percentage of red pixels
    red_pixels = np.count_nonzero(mask[exclude_black_pixels])
    total_pixels = np.count_nonzero(exclude_black_pixels)
    red_percentage = ((red_pixels / total_pixels) * 100)
    # print (red_percentage)
    return red_percentage

def get_green_percentages(img):
    """This function returns the total green percentage of the input image"""
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green hue roughly between 35° and 85°
    # (extend to include dark greens, based on what our data set has)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([90, 255, 255])

    # exclude very dark/black pixels
    exclude_black_pixels = hsv[:, :, 2] > 20 

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.count_nonzero(mask[exclude_black_pixels])
    total_pixels = np.count_nonzero(exclude_black_pixels)
    green_percentage = (green_pixels / total_pixels) * 100
    return green_percentage

def get_yellow_percentages(img):
    """This function returns the total yellow percentage of the input image"""
    if img is None:
        return None
    yellow_percentage = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow (lower saturation threshold to include pale yellows, as the yellow coloration in our data set is alomst white)
    lower_yellow = np.array([15, 20, 120])
    upper_yellow = np.array([40, 255, 255])
    
    exclude_black_pixels = hsv[:, :, 2] > 20  # exclude very dark/black pixels

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = np.count_nonzero(mask[exclude_black_pixels])
    total_pixels = np.count_nonzero(exclude_black_pixels)
    yellow_percentage = ((yellow_pixels / total_pixels) * 100)

    return yellow_percentage


def get_LBP(img):
    """This function calculates the LBP in a histogram of 8 bins and returns the data, the histogram en the bins"""
    if img is None:
        print(f"Could not read image by get_LBP")
        return
    
    # P = Number of circularly symmetric neighbor points. (8 tested relatively the best in general)
    P = 8
    # R = Radius of circle.
    R = 1

    # Converte to gray
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LBP_data = local_binary_pattern(grayed, P, R, method='uniform')

    # Compute histogram (to be able te return the usefull numbers)
    n_bins = int(LBP_data.max() + 1)
    hist, bins = np.histogram(LBP_data.ravel(), bins=n_bins, range=(0, n_bins))

    # print("LBP_data:")
    # print(LBP_data)
    # print("\nhist:")
    # print(hist)
    # print("\nbins:")
    # print(bins)
    # plot_LBP_results(img, LBP_data, hist, bins) # for visual testing

    return LBP_data, hist, bins

def get_fourth_element_LBP(img):
    """This function calls the LBP function and gets the value of the 4th bin and returns it <Zie verslag voor de redenen> """
    lbp_output = get_LBP(img)

    if lbp_output is None:
        print("LBP output is None")
        return None
        
    _, hist, _ = lbp_output
        
    if len(hist) < 4:
        print("Histogram has less than 4 bins")
        return None
    
    # 4th element (0-indexed)
    return hist[3]  

def get_eighth_element_LBP(img):
    """This function calls the LBP function and gets the value of the 8th bin and returns it <Zie verslag voor de redenen> """
    lbp_output = get_LBP(img)

    if lbp_output is None:
        print("LBP output is None")
        return None
        
    _, hist, _ = lbp_output
        
    if len(hist) < 8:
        print("Histogram has less than 4 bins")
        return None
    # 8th element (0-indexed)
    return hist[7]  


def plot_LBP_results(img, LBP_data, hist, bins):
    """Displays the original image, LBP result, and histogram with auto-naming."""

    # this plotting function was used to get the most important values of bins to then only aad the useful ones to the data via Add_to_dataframe.
    global _kind_index, _summary_data

    # Cycle through the KINDS list
    name = KINDS[_kind_index % len(KINDS)]
    _kind_index += 1  # increment for next call

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"Original Image - {name}")
    axs[0].axis("off")
    axs[1].imshow(LBP_data, cmap='gray')
    axs[1].set_title(f"LBP Image - {name}")
    axs[1].axis("off")
    axs[2].bar(bins[:-1], hist, width=0.7, color='gray', edgecolor='black')
    axs[2].set_title(f"LBP Histogram - {name}")
    axs[2].set_xlabel("LBP Value")
    axs[2].set_ylabel("Frequency")

    plt.tight_layout()
    # plt.show()
    
    #Store data for summary plot ---
    mean_val = np.mean(hist)
    var_val = np.var(hist)
    _summary_data.append((name, mean_val, var_val, hist, bins))

    # After all 6 kinds are processed ---
    if len(_summary_data) == len(KINDS):
        plot_summary_scatter(_summary_data)
        plot_all_histograms(_summary_data)

        # Reset for next run
        _summary_data.clear()
        _kind_index = 0



def plot_summary_scatter(summary_data):
    """Plot a scatter showing mean vs variance for each images LBP histogram."""
    names = [d[0] for d in summary_data]
    means = [d[1] for d in summary_data]
    vars_ = [d[2] for d in summary_data]

    plt.figure(figsize=(8, 6))
    plt.scatter(means, vars_, color='teal')

    for i, name in enumerate(names):
        plt.text(means[i] + 0.02, vars_[i], name, fontsize=9)

    plt.title("LBP Histogram Summary (Mean vs Variance)")
    plt.xlabel("Mean of Histogram")
    plt.ylabel("Variance of Histogram")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



def plot_all_histograms(summary_data):
    """Plot all LBP histograms together for comparison <Zie verslag>."""
    plt.figure(figsize=(10, 6))
    for name, _, _, hist, bins in summary_data:

        plt.plot(bins[:-1], hist, label=name, linewidth=1.5)

    plt.title("LBP Histogram Comparison Across Images")
    plt.xlabel("LBP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()
