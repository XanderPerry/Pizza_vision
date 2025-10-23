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

import pizza_cutter
import test_random
import get_values_x
import get_values_h

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
MODES = ["random", "test", "validation"]

Hayans_Path = "C:\HU\Jaar3\A\Beeldherkening\Pizza_vision\pizza_dataframes\Pizza15.csv"
Xanders_Path = "pizza_dataframes\Pizza15.csv"

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
        "circles_m" :           get_values_x.get_med_circles(img),
        "blobcount_s" :         get_values_x.get_blobcount_s(img),
        "blobcount_m" :         get_values_x.get_blobcount_m(img),
        "blobcount_l" :         get_values_x.get_blobcount_l(img)
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

def rf_predict(img, is_cutout=False):

    if not is_cutout:
        img = pizza_cutter.cut_pizza(img)

    new_sample = process_img(img)
   
    new_prediction_encoded = rf_model.predict(new_sample)

    predicted_kind = le.inverse_transform(new_prediction_encoded)[0]

    return predicted_kind

def test_model_rf(dataset="data", datagroup="validation"):
    label_true = []
    label_predicted = []

    if dataset == "data":
        is_cutout = False
    else:
        is_cutout = True
    
    for kind in KINDS:
        for filename in glob.glob(dataset + "/" + kind + "/" + datagroup + "/**/*.jpg", recursive=True):
            print(filename)
            label_true.append(kind)
            img = cv2.imread(filename)
            label_predicted.append(rf_predict(img, is_cutout=is_cutout))                
        
    accuracy = accuracy_score(label_true, label_predicted)
    print("Accuracy: "+str(accuracy))

    return label_true, label_predicted

# def train_knn(df_local_path):
#     pizza_df = pd.read_csv(df_local_path)

#     X = pizza_df.iloc[:, 2:].values
#     y = pizza_df["kind"].values

#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#     # scaler = StandardScaler()
#     # x_train = scaler.fit_transform(x_train)
#     # x_test = scaler.transform(x_test)
    
#     highest_accuracy = 0
#     best_n = 1
#     scores = []

#     print("test")
#     for i in range(1, 31):
#         knn = KNeighborsClassifier(n_neighbors=i)
#         score = cross_val_score(knn, X, y, cv=5)
#         scores.append(np.mean(score))

#         # knn.fit(x_train, y_train)

#         # y_pred = knn.predict(x_test)
#         # accuracy = accuracy_score(y_test, y_pred)

#         # if accuracy >= highest_accuracy:
#         #     accuracy = highest_accuracy
#         #     best_n = i

#         # knn = KNeighborsClassifier(n_neighbors=best_n)
#         # knn.fit(x_train, y_train)

#     sns.lineplot(x = range(1,31), y = scores, marker = 'o')
#     plt.xlabel("K Values")
#     plt.ylabel("Accuracy Score")
#     plt.show()

#     return knn

# def knn_predict(img_path):
#     new_sample = process_img(cv2.imread(img_path))

#     predicted_kind = knn.predict(new_sample)

#     return predicted_kind


# def test_model_knn():
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

def random_func():
     while True:
        print("Press esc to exit, press another key for new images.")

        imgs = test_random.get_random_images(dataset="data", datagroup="validation")
        predictions = test_random.apply_function(rf_predict, imgs)
        
        test_random.imgs_print_results(results_list=[predictions], labels_list=["Predictions"])

        imgs_res = {}
        for kind in KINDS:
            imgs_res[kind] = cv2.resize(imgs[kind], (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

            if kind == predictions[kind]:
                result = "Correct :)"
                color = (0, 255, 0)
            else:
                result = "Incorrect :("
                color = (0, 100, 255)

            text = "True: " + kind + " | Predicted: " + predictions[kind] + " | " + result
            imgs_res[kind] = cv2.putText(imgs_res[kind], text, (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            0.8, color, 1, cv2.LINE_AA)

        top_img = np.concatenate((imgs_res["che"], imgs_res["fun"], imgs_res["haw"]), axis = 1)
        bot_img = np.concatenate((imgs_res["mar"], imgs_res["moz"], imgs_res["sal"]), axis = 1)
        res_img = np.concatenate((top_img, bot_img), axis=0)

        cv2.imshow("Results", res_img)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key& 0xFF == 27:
            print("Exiting...")
            break
        else:
            cv2.destroyAllWindows()
            print("New images.")

if __name__ == "__main__": 
    print("Starting model training...")
    rf_model, le = train_rf(Xanders_Path)
    print("Model trained!\n")

    print("Welcome to the PizzaVision demo program, please choose one of the following modes: "+str(MODES))

    while True:
        mode = input("What mode do you want to run in? ").lower()
        while not mode in MODES:
            mode = input("This mode is unsupported, please try again: ")

        if mode == "random":
           random_func()

        elif mode == "test":
            label_true, label_predicted = test_model_rf(dataset="data", datagroup="test")
            plot_cm(label_true=label_true, label_predicted=label_predicted, labels=KINDS, title="Confusion matrix - RF")

        elif mode == "validation":
            label_true, label_predicted = test_model_rf(dataset="data", datagroup="validation")
            plot_cm(label_true=label_true, label_predicted=label_predicted, labels=KINDS, title="Confusion matrix - RF")

        if input("would you like another test?(y/n) ").lower() in ["y", "yes"]:
            continue
        else:
            break

    # knn = train_knn(Xanders_Path)
    # label_true, label_predicted = test_model_knn()
    # plot_cm(label_true=label_true, label_predicted=label_predicted, labels=KINDS, title="Confusion matrix - knn")
    
