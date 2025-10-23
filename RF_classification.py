##                               //this script a prototype of our first engine\\

##########################################################################################################
## Part 1:
## Takes in the data from our training data-set in the format of .csv file that were pre-cereated (the results of individual test)
## and trains the RF-model with it.
#####################################################

#####################################################
## Part 2:
## Takes in the images from our test-set and applies all the pre-processing filters and applies all the tests that we do,
## resluting in a the numirical values of all the tests (simular to the ones in DF).
#####################################################

#####################################################
## Part 3
## Takes the values from Part2 and tries and classify the images into their sorts using one or more of our descision engine.
#####################################################

#####################################################
## Part 4
## Gives the results.
#####################################################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from io import StringIO
import msvcrt
import time
import cv2
import get_values_x
import get_values_h
Hayans_Path = "C:\HU\Jaar3\A\Beeldherkening\Pizza_vision\pizza_dataframes\Pizza12.csv"
# "C:/HU/Jaar3/A/Beeldherkening/data_cutout/moz/test/moz_0_A_00_Y_0000.jpg"           ## test images pathing
Xanders_Path = "pizza_dataframes\Pizza12.csv"


def Train_RF_with_DF(DF_local_Path, img_path=None):
    
# --- 1. DATA SIMULATION (Using your CSV structure) ---
# Load the data into a Pandas DataFrame

    df = pd.read_csv(DF_local_Path)

    print("--- 1. Data Loaded from CSV (Pandas DataFrame) ---")
    print(df.head())
    print(f"\nTarget classes found: {df['kind'].unique()}")

    # --- 2. DATA PREPROCESSING (Feature and Label Separation) ---

    # The target variable (what we want to predict)
    # 'kind' column acts as the image recognition label (che, app, ban)
    Y = df['kind']

    # Features (all columns except 'ID' and 'kind')
    # These are the descriptive numerical features extracted from the images
    X = df.drop(['ID', 'kind'], axis=1)

    # Ensure the features are all numeric (critical for Random Forest)
    X = X.astype(float)

    # Encode the categorical target variable (Y) into numerical form
    # Random Forest can handle this, but scikit-learn generally expects numeric targets
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    class_names = le.classes_
    print(f"\nLabels Encoded: {le.classes_}")

    # --- 3. TRAIN/TEST SPLIT (Decision: How to evaluate the model) ---

    # Split the data into 80% for training and 20% for testing (unseen data)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
    )

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # --- 4. RANDOM FOREST DECISION ENGINE ---

    # Define the Random Forest Model (The Decision Engine)
    # n_estimators: Number of trees in the forest (more usually means better, but slower)
    # random_state: For reproducibility
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model (The engine learns the decision boundaries from the data)
    print("\n--- 4. Training Random Forest Model ---")
    rf_model.fit(X_train, Y_train)
    print("Training complete.")

    # --- 5. EVALUATION AND PREDICTION ---

    # 5.1 Predict on the unseen test data
    Y_pred = rf_model.predict(X_test)

    # 5.2 Evaluate performance
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"\n--- 5. Evaluation ---")
    print(f"Accuracy on Test Data: {accuracy:.2f}")

    # 5.3 Classification Report (Shows performance per class)
    report = classification_report(
        Y_test, Y_pred,
        target_names=class_names,
        zero_division=0 # Set to 0 to avoid warnings when a class has no samples in the test set
    )
    print("\nClassification Report:\n", report)

    # --- 6. MAKING A NEW PREDICTION (The final decision) ---

    if(img_path):
        new_sample = process_img(cv2.imread(img_path))
    else:
        return

    # Use the model to predict the class
    new_prediction_encoded = rf_model.predict(new_sample)

    # Convert the numerical prediction back to the readable class label
    predicted_kind = le.inverse_transform(new_prediction_encoded)[0]

    print("\n--- 6. Final Decision for a New Image ---")
    print(f"New Image Features:\n{new_sample.iloc[0].to_dict()}")
    print(f"\nPredicted Kind (Decision): {predicted_kind}")

    return

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

def main():
    # C:/HU/Jaar3/A/Beeldherkening/data_cutout/" + kind + "/train/
    Train_RF_with_DF(Hayans_Path, "C:/HU/Jaar3/A/Beeldherkening/data_cutout/moz/test/moz_0_A_00_Y_0000.jpg")
    print("Press ESC to stop the program.")

    while True:
        if msvcrt.kbhit():  # Check if a key was pressed
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC key
                print("ESC pressed. Exiting program...")
                break

if __name__ == "__main__":
    main()
