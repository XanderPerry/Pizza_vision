##                               //this script consists of 4 main parts\\

##########################################################################################################
## Part 1:
## Takes in the data from our training data-set in the format of .csv file that were pre-cereated (the results of individual test)
## and trains the RF-model with it.
#####################################################

#####################################################
## Part 2:
## Takes in the images from our test-set and applies all the pre-processing filters and applies all the tests that we do,
## resluting in a the numiraical values of all the tests.
#####################################################

#####################################################
## Part 3
## Takes the values from Part2 and trys and classify the images into their sorts.
#####################################################

#####################################################
## Part 4
## Gives the results.
#####################################################
##########################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score





##################################################### Part 1: begin

# --- 1. DATA SIMULATION (Using your CSV structure) ---
#C:\HU\Jaar3\A\Beeldherkening\Pizza_vision\pizza_dataframes\Pizza10.csv
# Load the data into a Pandas DataFrame

from io import StringIO
df = pd.read_csv("C:\HU\Jaar3\A\Beeldherkening\Pizza_vision\pizza_dataframes\Pizza10.csv")

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

# Define new, unseen features (e.g., from a new image)
# This new sample looks like a 'che' (low hue, high sat, high LBP)
new_sample = pd.DataFrame([{ # che72
    'mean_hue': 17.44428853232465, 'mean_sat': 115.95940861855082, 'mean_val': 1141.27171415207533, 
    'edge_percent': 16.42568075976654, 'fourth_LBP': 8252, 'eighth_LBP': 1538, 
    'Red percentage': 7.410500171291538, 'Yellow percentage': 46.22302158273381, 
    'Green percentage': 0.1006337786913326, 'circles_s': 3.0, 'circles_m': 1.0
}])

# Use the model to predict the class
new_prediction_encoded = rf_model.predict(new_sample)

# Convert the numerical prediction back to the readable class label
predicted_kind = le.inverse_transform(new_prediction_encoded)[0]

print("\n--- 6. Final Decision for a New Image ---")
print(f"New Image Features:\n{new_sample.iloc[0].to_dict()}")
print(f"\nPredicted Kind (Decision): {predicted_kind}")



##################################################### Part 1: end






##################################################### Part 2: begin



##################################################### Part 2: end

##################################################### Part 3: begin



##################################################### Part 3: end

##################################################### Part 4: begin



##################################################### Part 4: end



