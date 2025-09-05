# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 16:12:54 2025

@author: Xande
"""
import cv2
import os

# Settings
train_count = 250
test_count = 50
image_width = 640
image_height = 480

pizza_type = input("Which pizza are you going to photograph? ")

save_dir_train = f"data/{pizza_type}/train"
save_dir_test = f"data/{pizza_type}/test"

# Create directories if they don't exist
os.makedirs(save_dir_train, exist_ok=True)
os.makedirs(save_dir_test, exist_ok=True)

pizza_types = ["mar", "sal", "che", "haw", "fun", "moz", "test"]
background_types = ["0", "1"]
light_types = ["A", "B"]
training = "X"
test = "Y"

if pizza_type in pizza_types:
    for background in background_types:
        for light in light_types:
            print("\nAbout to start capturing for parameters:")
            print(f"Pizza = {pizza_type}")
            print(f"Background = {background}")
            print(f"Light = {light}")
            input("Press any key to start")

            # Initialize webcam
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Error: Could not access the webcam.")
                exit()
            
            print("Starting image capture. Press 'q' to quit.")
            
            image_counter = 0
            total_images = train_count + test_count
            
            while image_counter < total_images:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
            
                # Resize the frame (optional)
                frame = cv2.resize(frame, (image_width, image_height))
            
                # Show the frame
                cv2.imshow("Image Capture (press 'q' to quit)", frame)
            
                # Save the image
                if image_counter < train_count:
                    filename = f"{pizza_type}_{background}_{light}_00_{training}_{image_counter}.jpg"
                    filepath = os.path.join(save_dir_train, filename)
                else:
                    filename = f"{pizza_type}_{background}_{light}_00_{test}_{image_counter-250}.jpg"
                    filepath = os.path.join(save_dir_test, filename)
            
                cv2.imwrite(filepath, frame)
                print(f"[{image_counter + 1}/{total_images}] Saved: {filepath}")
                image_counter += 1
            
                # Wait for 100ms between captures
                key = cv2.waitKey(100)
                if key & 0xFF == ord('q'):
                    print("Manual interruption by user.")
                    break
            
            print("Image capture completed.")
            cv2.destroyAllWindows()
            cap.release()
