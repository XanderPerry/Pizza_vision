import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Predefined image names
KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
_kind_index = 0  # internal counter to track which name to use next
_summary_data = []  # store (name, mean, variance) for scatter plot



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

    # print("LBP_data:")
    # print(LBP_data)
    # print("\nhist:")
    # print(hist)
    # print("\nbins:")
    # print(bins)
    plot_LBP_results(img, LBP_data, hist, bins)
    return LBP_data, hist, bins



def plot_LBP_results(img, LBP_data, hist, bins):
    """Displays the original image, LBP result, and histogram with auto-naming."""
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
    
    # --- Store data for summary plot ---
    mean_val = np.mean(hist)
    var_val = np.var(hist)
    _summary_data.append((name, mean_val, var_val, hist, bins))

    # --- After all 6 are processed ---
    if len(_summary_data) == len(KINDS):
        plot_summary_scatter(_summary_data)
        plot_all_histograms(_summary_data)

        # Reset for next run
        _summary_data.clear()
        _kind_index = 0



def plot_summary_scatter(summary_data):
    """Plot a scatter showing mean vs variance for each image’s LBP histogram."""
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
    """Plot all LBP histograms together for comparison."""
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
