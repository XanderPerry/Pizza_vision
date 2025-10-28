import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn import tree
import cv2
import glob

# import own .py files
import pizza_cutter
import test_random
import get_values_x
import get_values_h


KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]
# All features that can be gotten.
FEATURES = ["mean_hue", "mean_sat", "mean_val", "edge_percent", "fourth_LBP", "eighth_LBP", "Red percentage", "Yellow percentage", "Green percentage", "circles_s", "circles_m", "blobcount_s", "blobcount_m", "blobcount_l","cc_count", "cc_mean", "mean_des_sift", "n_sift", "cc_mean_area"]
# The possible user input.
MODES = ["live", "random", "test", "validation", "tree"]

Hayans_Path = "C:\HU\Jaar3\A\Beeldherkening\Pizza_vision\pizza_dataframes\Pizza19.csv"
Xanders_Path = "pizza_dataframes\Pizza19.csv"

# Get all features of image and set them in a DF as done in the .csv files. applying all preprocessing internally
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
        "blobcount_l" :         get_values_x.get_blobcount_l(img),
        "cc_count" :            get_values_x.get_cc_n(img),
        "cc_mean" :             get_values_x.get_cc_mean(img),
        "mean_des_sift" :       get_values_x.get_mean_des_sift(img),
        "n_sift" :              get_values_x.get_n_sift(img),
        "cc_mean_area" :        get_values_x.get_cc_mean_area(img)
    }])

    return df

# Train the RF.model with the latest .csv files that are based on the training data set. (latest .csv being, Pizza19.csv)
def train_rf(df_local_path):
    # Read the dataset from the .csv file.
    df = pd.read_csv(df_local_path)

    # Extract the ID, Kind and the Kind for the label
    Y = df["kind"]
    X = df.drop(['ID', 'kind'], axis=1)

    # Ensure all feature values are of type float (for compatibility with the model)
    X = X.astype(float)

    # Initialize a LabelEncoder to convert string labels into numeric values
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)

            #################################
            #
            #  inspired by AI
            #
            #################################   

    # Split the dataset into training and testing sets
    # - 90% training data, 10% testing data
    # - random_state=42 ensures reproducibility
    # - stratify=Y_encoded ensures the class distribution is preserved across sets
    # to calculate an initial accuracy.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_encoded, test_size=0.1, random_state=42, stratify=Y_encoded
    )
            #################################
            #  End of AI-inspired code
            #################################

     # Create a Random Forest classifier with 100 decision trees and fix the seed of the random number generator with 42
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest model using the training data of the split function.
    rf_model.fit(X_train, Y_train)

    # Use the trained model to predict labels for the test data of the split function.
    Y_pred = rf_model.predict(X_test)

    # Calculate the accuracy score of the model’s predictions (Only the initail accuracy of the split functiona NOT of the CM)
    accuracy = accuracy_score(Y_test, Y_pred)

    print("Predicted training accuracy: " + str(accuracy))

    # Return the trained Random Forest model and the label encoder
    # (the encoder is useful later to decode predicted label numbers back to their original string names)
    return rf_model, le

def rf_predict(img, is_cutout=False):
    """A function used to an image (original/cutout) and applies all preprocessing on it, extracts all its features and preditcts its present pizza kind."""
    if not is_cutout:
        img = pizza_cutter.cut_pizza(img)

    new_sample = process_img(img)
   
    new_prediction_encoded = rf_model.predict(new_sample)

    # revert back to the name.
    predicted_kind = le.inverse_transform(new_prediction_encoded)[0]

    return predicted_kind

def test_model_rf(dataset="data", datagroup="validation"):
    
    """This function tests the predictions of the model on all images input (validation/test/train) then compairs the results to the correct kinds and outputs its accuracy."""
    # it also returns the prediction and the true labels for them to be used for the CM in the main.

    label_true = []
    label_predicted = []

    if dataset == "data":
        is_cutout = False
    else:
        is_cutout = True
    
    for kind in KINDS:
        for filename in glob.glob(dataset + "/" + kind + "/" + datagroup + "/**/*.jpg", recursive=True):
        # for filename in glob.glob("C:/HU/Jaar3/A/Beeldherkening/data set/data/" + kind + "/" + datagroup + "/**/*.jpg", recursive=True):
            print(filename)
            label_true.append(kind)
            img = cv2.imread(filename)

            # Get prediction
            label_predicted.append(rf_predict(img, is_cutout=is_cutout))                
    # Get accuracy by comparing prediction to true.
    accuracy = accuracy_score(label_true, label_predicted)
    print("Accuracy: "+str(accuracy))

    # return both lists for later use.
    return label_true, label_predicted    

def plot_cm(label_true, label_predicted, labels=None, title="Confusion matrix"):
    """This function takes lists of the predictions and true and outputs a confusion matrix."""
   
    cm = confusion_matrix(label_true, label_predicted, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def live_loop():
    
    """This function allows the user to capture a photo live with a webcam and then makes a prediction on the kind of pizza in it"""
    image_width = 640
    image_height = 480
    
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()
    
    print("Starting image capture. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
            
        # resize the image to the standerd used in this project.
        frame = cv2.resize(frame, (image_width, image_height))
    
        cv2.imshow("Live feed (press 'q' to quit, 'n' to capture new frame)", frame)

        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'):
            print("Manual interruption by user.")
            break
        elif key & 0xFF == ord('n'):
            unused, img = cap.read()
            cv2.imshow("Captured image", img)

            # make the predicion with the trained model.
            prediction = rf_predict(img)
            text = "Prediction: " + prediction
            img = cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("Captured image", img)

    cv2.destroyAllWindows()
    cap.release()

def random_loop():
     """This function randomly choses 6 images off of the validation set (one of each pizza kind) and makes a predicion on each on them.
      then checks if the prediction is correct and prints the results on the screen in addition to showing the chosen images ."""
     
     while True:
        print("Press 'q' to exit, press another key for new images.")

        load_img = np.zeros((int(480*2*0.7), int(640*3*0.7), 3), dtype = np.uint8)
        load_img = cv2.putText(load_img, "Loading...", (400, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Results", load_img)
        key = cv2.waitKey(1)

        imgs = test_random.get_random_images(dataset="data", datagroup="validation")
        predictions = test_random.apply_function(rf_predict, imgs)
        
        test_random.imgs_print_results(results_list=[predictions], labels_list=["Predictions"])

        imgs_res = {}
        for kind in KINDS:
            imgs_res[kind] = cv2.resize(imgs[kind], (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

            # check if the prediction was correct and print the results anyway.
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
        if key & 0xFF == ord('q'):
            print("Exiting...")
            break
        else:
            print("New images.")

def show_tree():
    """This function shows a visualization of the working of RF. which is very complix to read but its used to prove a point in the report <het verslag>"""
    new_tree = True
    while new_tree:
        plt.figure(figsize=(25,20))
        _ = tree.plot_tree(rf_model.estimators_[0],
                        feature_names=FEATURES,
                        class_names=KINDS,
                        filled=True)
        plt.show()

        if input("would you like another plot?(y/n) ").lower() in ["y", "yes"]:
            new_tree = True
        else:
            new_tree = False
        


"""take the user input to chose the mode of running."""
if __name__ == "__main__": 
    print("Starting model training...")
    rf_model, le = train_rf(Xanders_Path)
    print("Model trained!\n")

    print("Welcome to the PizzaVision demo program, please choose one of the following modes: "+str(MODES))

    while True:
        mode = input("What mode do you want to run in? ").lower()
        while not mode in MODES:
            mode = input("This mode is unsupported, please try again: ")

        if mode == "live":
            live_loop()

        elif mode == "random":
            random_loop()

        elif mode == "test":
            label_true, label_predicted = test_model_rf(dataset="data", datagroup="test")
            plot_cm(label_true=label_true, label_predicted=label_predicted, labels=KINDS, title="Confusion matrix - RF")

        elif mode == "validation":
            label_true, label_predicted = test_model_rf(dataset="data", datagroup="validation")
            plot_cm(label_true=label_true, label_predicted=label_predicted, labels=KINDS, title="Confusion matrix - RF")

        elif mode == "tree":
            show_tree()

        if input("would you like another test?(y/n) ").lower() in ["y", "yes"]:
            continue
        else:
            break
    
