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

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]

pizza_df = pd.DataFrame(columns=["kind", "hue_dist", "mean_hue", "mean_sat", "mean_val", "edge_percentage"])

for kind in KINDS:
    print(f"Processing {kind} pizzas...")
    i = 0
    for filename in glob.glob("data/" + kind + "/train/**/*.jpg", recursive=True):
        i += 1
        if i % 100 != 0:
            continue

        # # Load image
        print(filename)
        img = cv2.imread(filename)
        # cv2.imshow("Original image", img)

        # Crop image based on color mask (red/yellow)
        img_cropped = pizza_cutter.crop_image(img)
        # cv2.imshow("Cropped image", img_cropped)

        # Crop image to circle using Hough Transform
        # img_circle_cropped = pizza_cutter.cutout_circle(img_cropped, edge_detection="sobel")
        # cv2.imshow("Detected circle- cropped", img_circle_cropped)

        img_circle = pizza_cutter.cutout_circle(img, edge_detection="sobel")
        # cv2.imshow("Detected circle", img_circle)

        # Plot RGB and HSV histograms
        # pizza_plotter.plot_rgb_histogram(img_circle) 
        # pizza_plotter.plot_hsv_histogram(img_circle)


        # Fill in dataframe with median hue, saturation, and value
        pizza_df.loc[len(pizza_df)] = [
            kind,
            pizza_plotter.get_hue_distribution(img_circle),
            pizza_plotter.get_mean_hue(img_circle),
            pizza_plotter.get_mean_sat(img_circle),
            pizza_plotter.get_mean_val(img_circle),
            pizza_plotter.get_edge_percentage(img_circle)
        ]

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            print("Manual interruption by user.")
            break
        elif key & 0xFF == 27:
            cv2.destroyAllWindows()
            print("Exiting...")
            exit(0)

        cv2.destroyAllWindows()

categories = np.unique(pizza_df["kind"])
colors = np.linspace(0, 1, len(categories))
colordict = dict(zip(categories, colors))
pizza_df["Color"] = pizza_df["kind"].apply(lambda x: colordict[x])

# pizza_df.plot.hist(column="edge_percentage", bins=30, by="kind", alpha=0.5, legend=True)
# plt.title("Edge percentage distribution by pizza type")
# plt.xlabel("Edge percentage")
# plt.ylabel("# of images")
# plt.show()

pizza_df.plot.scatter(x='mean_hue', y='edge_percentage', c='Color', colormap='jet')
plt.title("Pizza types by mean hue and edge percentage")
plt.xlabel("Mean Hue")
plt.ylabel("Edge Percentage")
plt.show()

plt.figure(figsize=(10, 6))
for kind in KINDS:
    subset = pizza_df[pizza_df["kind"] == kind]["mean_hue"]
    plt.hist(subset, bins=30, alpha=0.5, label=kind)
plt.title("Mean hue distribution for all pizza types")
plt.xlabel("Mean Hue")
plt.ylabel("# of images")
plt.legend()
plt.show()