import numpy as np
import glob
import matplotlib.pyplot as plt

import cv2

def plot_color_histogram(cropped_image):
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges
    lower_yellow = np.array([20, 50, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_red1 = np.array([0, 50, 100])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 50, 100])
    upper_red2 = np.array([179, 255, 255])

    # Create masks
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Count pixels
    yellow_pixels = cv2.countNonZero(mask_yellow)
    red_pixels = cv2.countNonZero(mask_red)
    total_pixels = cropped_image.shape[0] * cropped_image.shape[1]
    other_pixels = total_pixels - yellow_pixels - red_pixels

    # Plot histogram
    categories = ['Red', 'Yellow', 'Other']
    counts = [red_pixels, yellow_pixels, other_pixels]
    colors = ['red', 'gold', 'gray']

    plt.figure(figsize=(8, 5))
    plt.bar(categories, counts, color=colors)
    plt.title('Color Pixel Count in Cropped Pizza Region')
    plt.xlabel('Color')
    plt.ylabel('Pixel Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Main loop
i = 50
for filename in glob.glob("data/che/train/**/*.jpg", recursive=True):
    if i < 50:
        i += 1
        continue
    else:
        i = 0

    image = cv2.imread(filename)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define color ranges
    lower_yellow = np.array([20, 50, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_red1 = np.array([0, 50, 100])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 50, 100])
    upper_red2 = np.array([179, 255, 255])

    # Create masks
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_yellow, mask_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    last_cropped = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # Crop the detected region
            last_cropped = image[y:y+h, x:x+w]

    # Show detection result
    cv2.imshow("Pizza Detection", image)

    # If a valid cropped region was found, show it and plot histogram
    if last_cropped is not None:
        cv2.imshow("Cropped Pizza", last_cropped)
        plot_color_histogram(last_cropped)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print("Manual interruption by user.")
        break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
