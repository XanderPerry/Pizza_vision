#   This script uses a KNN classifier for pizza predictions, due to significantly inferior results compared to random forest
#       this file is UNUSED.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import cv2

import get_values_h
import get_values_x

def process_img(img):
    df = pd.DataFrame([{
        "mean_hue" :            get_values_x.get_mean_hue(img), 
        "mean_sat" :            get_values_x.get_mean_sat(img), 
        "mean_val" :            get_values_x.get_mean_val(img), 
        "edge_percent" :        get_values_x.get_edge_percentage(img), 
        "fourth_LBP" :          get_values_h.get_fourth_element_LBP(img), 
        "eighth_LBP" :          get_values_h.get_eighth_element_LBP(img), 
        "Red percentage" :      get_values_h.get_red_percentages(img), 
        "Yellow percentage" :   get_values_h.get_yellow_percentages(img), 
        "Green percentage" :    get_values_h.get_green_percentages(img), 
        "circles_s" :           get_values_x.get_small_circles(img), 
        "circles_m" :           get_values_x.get_med_circles(img)
    }])

    return df

    # list = []
    # list.append(get_values_x.get_mean_hue(img))
    # list.append(get_values_x.get_mean_sat(img))
    # list.append(get_values_x.get_mean_val(img))
    # list.append(get_values_x.get_edge_percentage(img))
    # list.append(get_values_h.get_fourth_element_LBP(img))
    # list.append(get_values_h.get_eighth_element_LBP(img))
    # list.append(get_values_h.get_red_percentages(img))
    # list.append(get_values_h.get_yellow_percentages(img))
    # list.append(get_values_h.get_green_percentages(img))
    # list.append(get_values_x.get_small_circles(img))
    # list.append(get_values_x.get_med_circles(img))
    
    # return [list]

pizza_df = pd.read_csv("pizza_dataframes/Pizza12.csv")

x = pizza_df.iloc[:, 2:].values
y = pizza_df["kind"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# for i in range(1, 200, 10):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(x_train, y_train)

#     y_pred = knn.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy " + str(i) + ": ", accuracy)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: " + str(accuracy))

print("prediction: ")
print(knn.predict(process_img(cv2.imread("data_cutout/moz/test/moz_1_B_00_Y_0075.jpg"))))
