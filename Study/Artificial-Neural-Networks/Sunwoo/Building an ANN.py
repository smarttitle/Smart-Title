# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(p=0.1))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting classifier to the Training set

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making the predictions and evaluating the model

# HW_X_test = np.ndarray(shape=(1,11), dtype=float)
# HW_X_test[0] = X_test[1000]

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


""" Predict if the customer with the following informations will leave the bank
Geography : France
Credit Score : 600
Gender : Male
Age : 40
Tenure : 3
Balance : 60000
Number of Products : 2
Has Credit Card : Yes
Is Active Member : Yes
Estimate Salary : 50000
"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
