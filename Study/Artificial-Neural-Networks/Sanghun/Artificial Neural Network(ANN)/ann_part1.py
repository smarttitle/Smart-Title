# Artificial Neural Network

# Part 1 - Data preprocessing

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
Xlabelencoder_country = LabelEncoder()
X[:, 1] = Xlabelencoder_country.fit_transform(X[:, 1])
Xlabelencoder_gender = LabelEncoder()
X[:, 2] = Xlabelencoder_gender.fit_transform(X[:, 2])
# Create dummy variable
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
'''
To center the data (make it have zero mean and unit standard error),
you subtract the mean and then divide the result by the standard deviation.
You do that on the training set of data. But then you have to apply the same
transformation to your testing set (e.g. in cross-validation), or to newly
obtained examples before forecast.
But you have to use the same two parameters μ and σ (values) that you used for
centering the training set.
Hence, every sklearn's transform's fit() just calculates the parameters
(e.g. μ and σ in case of StandardScaler) and saves them as an internal
objects state. Afterwards, you can call its transform() method to apply the
transformation to a particular set of examples fit_transform() joins
these two steps and is used for the initial fitting of parameters
on the training set x, but it also returns a transformed x′.
Internally, it just calls first fit() and then transform() on the same data.
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making Artificial Neural Network

# Importing Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(kernel_initializer='uniform', activation='relu', input_dim=11, output_dim=6))

# Adding the second hidden layer
classifier.add(Dense(kernel_initializer='uniform', activation='relu', output_dim=6))

# Adding output layer
classifier.add(Dense(kernel_initializer='uniform', activation='sigmoid', output_dim=1))

# Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Model Prediction and evaluation
predictions = classifier.predict(X_test)
predictions = (predictions > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

'''
<Predict new observation>
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction)