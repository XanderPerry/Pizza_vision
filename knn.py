from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

import get_values_h
import get_values_x

pizza_df = pd.read_csv("pizza_dataframes/Pizza11.csv")

x = pizza_df.iloc[:, 2:].values
y = pizza_df["kind"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("Unscaled:")
print(x_train.head())
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

print("Scaled:")
print(x_train.head())

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
