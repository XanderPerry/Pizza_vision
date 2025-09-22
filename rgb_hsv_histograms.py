##################################
# RGB and HSV color histograms  #
##################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import cv2

import pizza_cutter
import pizza_plotter
import whitebalance

pizza_df = pd.DataFrame(columns=["kind", "hue_dist", "median_hue", "median_sat", "median_val"])

for kind in ["che", "fun", "haw", "mar", "moz", "sal"]:
    print(f"Processing {kind} pizzas...")
    i = 0
    for filename in glob.glob("data/" + kind + "/train/**/*.jpg", recursive=True):
        # # Load image
        img = cv2.imread(filename)
        # cv2.imshow("Original image", img)

        # # Crop image based on color mask (red/yellow)
        img_cropped = pizza_cutter.crop_image(img)
        # cv2.imshow("Cropped image", img_cropped)

        # # Crop image to circle using Hough Transform
        img_circle = pizza_cutter.cutout_circle(img, edge_detection="sobel")
        # cv2.imshow("Detected circle", img_circle)

        # Plot RGB and HSV histograms
        # pizza_plotter.plot_rgb_histogram(img_circle) 
        # pizza_plotter.plot_hsv_histogram(img_circle)

        # Fill in dataframe with median hue, saturation, and value
        pizza_df.loc[len(pizza_df)] = [
            kind,
            pizza_plotter.get_hue_distribution(img_circle),
            pizza_plotter.get_median_hue(img_circle),
            pizza_plotter.get_median_sat(img_circle),
            pizza_plotter.get_median_val(img_circle)
        ]

        # key = cv2.waitKey(0)
        # if key & 0xFF == ord('q'):
        #     print("Manual interruption by user.")
        #     break

        i += 1
        if i > 10:
            break
        cv2.destroyAllWindows()

# categories = np.unique(pizza_df["kind"])
# colors = np.linspace(0, 1, len(categories))
# colordict = dict(zip(categories, colors))
# pizza_df["Color"] = pizza_df["kind"].apply(lambda x: colordict[x])
# pizza_df.plot.scatter(x='median_hue', y='median_sat', c='Color', colormap='viridis')
# plt.title("Pizza types by median hue and saturation")
# plt.xlabel("Median Hue")
# plt.ylabel("Median Saturation")
# plt.show()