import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import cv2
import glob

import get_values_x
import get_values_h

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]

Hayans_Path = "C:\HU\Jaar3\A\Beeldherkening\Pizza_vision\pizza_dataframes\Pizza12.csv"
Xanders_Path = "pizza_dataframes\Pizza12.csv"

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

def train_rf(df_local_path):
    df = pd.read_csv(df_local_path)

    Y = df["kind"]
    X = df.drop(['ID', 'kind'], axis=1)
    X = X.astype(float)

    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    class_names = le.classes_

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_model.fit(X_train, Y_train)

    Y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    return rf_model, le

def rf_predict(img_path):
    new_sample = process_img(cv2.imread(img_path))
   
    new_prediction_encoded = rf_model.predict(new_sample)

    predicted_kind = le.inverse_transform(new_prediction_encoded)[0]

    return predicted_kind

def test_model_rf():
    label_true = []
    label_predicted = []
    
    for kind in KINDS:
        for filename in glob.glob("data_cutout/" + kind + "/test/**/*.jpg", recursive=True):
            print(filename)
            label_true.append(kind)
            label_predicted.append(rf_predict(filename))                
        
    accuracy = accuracy_score(label_true, label_predicted)
    print("Accuracy: "+str(accuracy))

    return label_true, label_predicted

def train_knn(df_local_path):
    pizza_df = pd.read_csv(df_local_path)

    X = pizza_df.iloc[:, 2:].values
    y = pizza_df["kind"].values

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    
    highest_accuracy = 0
    best_n = 1
    scores = []

    print("test")
    for i in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(knn, X, y, cv=5)
        scores.append(np.mean(score))

        # knn.fit(x_train, y_train)

        # y_pred = knn.predict(x_test)
        # accuracy = accuracy_score(y_test, y_pred)

        # if accuracy >= highest_accuracy:
        #     accuracy = highest_accuracy
        #     best_n = i

        # knn = KNeighborsClassifier(n_neighbors=best_n)
        # knn.fit(x_train, y_train)

    sns.lineplot(x = range(1,31), y = scores, marker = 'o')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.show()

    return knn

def knn_predict(img_path):
    new_sample = process_img(cv2.imread(img_path))

    predicted_kind = knn.predict(new_sample)

    return predicted_kind


def test_model_knn():
    label_true = []
    label_predicted = []
    
    for kind in KINDS:
        for filename in glob.glob("data_cutout/" + kind + "/test/**/*.jpg", recursive=True):
            print(filename)
            label_true.append(kind)
            label_predicted.append(knn_predict(filename))                
        
    accuracy = accuracy_score(label_true, label_predicted)
    print("Accuracy: "+str(accuracy))

    return label_true, label_predicted
    

def plot_cm(label_true, label_predicted, labels=None, title="Confusion matrix"):
    cm = confusion_matrix(label_true, label_predicted, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == "__main__": 
    rf_model, le = train_rf(Xanders_Path)
    label_true, label_predicted = test_model_rf()
    plot_cm(label_true=label_true, label_predicted=label_predicted, labels=KINDS, title="Confusion matrix - RF")

    # knn = train_knn(Xanders_Path)
    # label_true, label_predicted = test_model_knn()
    # plot_cm(label_true=label_true, label_predicted=label_predicted, labels=KINDS, title="Confusion matrix - knn")
    
