import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern



def get_LBP(img):
    if img is None:
        print(f"Could not read image by get_LBP")
        return
    
    # P = Number of circularly symmetric neighbor points. (8 for the test only)
    P = 8
    # R = Radius of circle.
    R = 1
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LBP_data = local_binary_pattern(grayed, P, R, method='uniform')

    # Compute histogram (to be able te return the usefull numbers)
    n_bins = int(LBP_data.max() + 1)
    hist, bins = np.histogram(LBP_data.ravel(), bins=n_bins, range=(0, n_bins))

    print("LBP_data:")
    print(LBP_data)
    print("\nhist:")
    print(hist)
    print("\nbins:")
    print(bins)
    plot_LBP_results(img, LBP_data, hist, bins)
    return LBP_data, hist, bins



def plot_LBP_results(img, LBP_data, hist, bins):
    """Displays the original image, LBP result, and histogram."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(LBP_data, cmap='gray')
    axs[1].set_title("LBP Image")
    axs[1].axis("off")

    axs[2].bar(bins[:-1], hist, width=0.7, color='gray', edgecolor='black')
    axs[2].set_title("LBP Histogram")
    axs[2].set_xlabel("LBP Value")
    axs[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()