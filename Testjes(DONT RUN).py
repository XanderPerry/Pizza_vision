

# some tests on colos values 
# and lbp local_binary_pattern





# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def get_all_images(folder):
#     """Recursively get paths to all image files in the given folder."""
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
#     image_paths = []
#     for root, _, files in os.walk(folder):
#         for f in files:
#             if f.lower().endswith(image_extensions):
#                 image_paths.append(os.path.join(root, f))
#     return image_paths

# def extract_hsv_values(image_paths, sample_fraction=0.1):
#     """Extract HSV values from all images (optionally subsample pixels for speed)."""
#     h_vals, s_vals, v_vals = [], [], []

#     for path in image_paths:
#         img = cv2.imread(path)
#         if img is None:
#             print(f"⚠️ Could not read {path}")
#             continue
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#         # Flatten the HSV arrays
#         h, s, v = hsv[:, :, 0].flatten(), hsv[:, :, 1].flatten(), hsv[:, :, 2].flatten()

#         # Optionally sample to reduce memory
#         n = int(len(h) * sample_fraction)
#         idx = np.random.choice(len(h), n, replace=False)
#         h_vals.extend(h[idx])
#         s_vals.extend(s[idx])
#         v_vals.extend(v[idx])

#     return np.array(h_vals), np.array(s_vals), np.array(v_vals)

# def plot_hsv_distributions(h, s, v):
#     """Plot histograms of the H, S, and V values."""
#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 3, 1)
#     plt.hist(h, bins=180, color='r', alpha=0.7)
#     plt.title('Hue Distribution')
#     plt.xlabel('Hue (0–179)')
#     plt.ylabel('Frequency')

#     plt.subplot(1, 3, 2)
#     plt.hist(s, bins=256, color='g', alpha=0.7)
#     plt.title('Saturation Distribution')
#     plt.xlabel('Saturation (0–255)')

#     plt.subplot(1, 3, 3)
#     plt.hist(v, bins=256, color='b', alpha=0.7)
#     plt.title('Value (Brightness) Distribution')
#     plt.xlabel('Value (0–255)')

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     folder = input("Enter the root folder containing photos: ").strip()
#     print("🔍 Searching for images...")
#     image_paths = get_all_images(folder)
#     print(f"Found {len(image_paths)} image(s).")

#     if not image_paths:
#         print("❌ No images found. Exiting.")
#         exit()

#     print("🎨 Extracting HSV values...")
#     h, s, v = extract_hsv_values(image_paths, sample_fraction=0.05)
#     print(f"✅ Extracted HSV values from {len(image_paths)} images.")

#     print("📊 Plotting distributions...")
#     plot_hsv_distributions(h, s, v)








# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def get_all_images(folder):
#     """Recursively get paths to all image files in the given folder."""
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
#     image_paths = []
#     for root, _, files in os.walk(folder):
#         for f in files:
#             if f.lower().endswith(image_extensions):
#                 image_paths.append(os.path.join(root, f))
#     return image_paths

# def extract_hsv_values(image_paths, sample_fraction=0.01):
#     """Extract HSV and RGB values from all images (optionally subsample pixels)."""
#     hsv_points = []
#     rgb_points = []

#     for path in image_paths:
#         img = cv2.imread(path)
#         if img is None:
#             print(f"⚠️ Could not read {path}")
#             continue
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

#         h = hsv[:, :, 0].flatten()
#         s = hsv[:, :, 1].flatten()
#         v = hsv[:, :, 2].flatten()
#         rgb_flat = img.reshape(-1, 3)

#         n = int(len(h) * sample_fraction)
#         idx = np.random.choice(len(h), n, replace=False)

#         hsv_points.append(np.stack([h[idx], s[idx], v[idx]], axis=1))
#         rgb_points.append(rgb_flat[idx] / 255.0)  # normalize RGB for plotting

#     if not hsv_points:
#         return np.empty((0, 3)), np.empty((0, 3))

#     hsv_points = np.concatenate(hsv_points, axis=0)
#     rgb_points = np.concatenate(rgb_points, axis=0)

#     return hsv_points, rgb_points

# def plot_hsv_3d(hsv_points, rgb_points):
#     """Plot 3D scatter plot of HSV values colored by their corresponding RGB color."""
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')

#     h, s, v = hsv_points[:, 0], hsv_points[:, 1], hsv_points[:, 2]
#     ax.scatter(h, s, v, c=rgb_points, s=2, alpha=0.6)

#     ax.set_xlabel('Hue (0–179)')
#     ax.set_ylabel('Saturation (0–255)')
#     ax.set_zlabel('Value (0–255)')
#     ax.set_title('3D HSV Color Distribution')

#     plt.show()

# if __name__ == "__main__":
#     folder = input("Enter the root folder containing photos: ").strip()
#     print("🔍 Searching for images...")
#     image_paths = get_all_images(folder)
#     print(f"Found {len(image_paths)} image(s).")

#     if not image_paths:
#         print("❌ No images found. Exiting.")
#         exit()

#     print("🎨 Extracting HSV values...")
#     hsv_points, rgb_points = extract_hsv_values(image_paths, sample_fraction=0.01)
#     print(f"✅ Extracted {len(hsv_points)} color samples.")

#     print("📊 Plotting 3D color scatter...")
#     plot_hsv_3d(hsv_points, rgb_points)








# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.feature import local_binary_pattern

# def plot_image_lbp(image_path, P=8, R=1):
#     """
#     Computes and plots the Local Binary Pattern (LBP) of an image.
    
#     Parameters:
#         image_path (str): Path to the image file.
#         P (int): Number of circularly symmetric neighbor points.
#         R (float): Radius of circle.
#     """
#     # Read and convert to grayscale
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"⚠️ Could not read image: {image_path}")
#         return

#     # Compute LBP
#     lbp = local_binary_pattern(img, P, R, method='uniform')

#     # Compute histogram of LBP values
#     n_bins = int(lbp.max() + 1)
#     hist, bins = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

#     # Plot results
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     ax1, ax2, ax3 = axes

#     ax1.imshow(img, cmap='gray')
#     ax1.set_title('Original Image')
#     ax1.axis('off')

#     ax2.imshow(lbp, cmap='gray')
#     ax2.set_title('Local Binary Pattern')
#     ax2.axis('off')

#     ax3.bar(bins[:-1], hist, width=0.8, color='gray')
#     ax3.set_title('LBP Histogram')
#     ax3.set_xlabel('LBP Value')
#     ax3.set_ylabel('Frequency')

#     plt.tight_layout()
#     plt.show()

