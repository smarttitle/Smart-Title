# Improving ANN
# dropout regularization to reduce overfitting if needed
# add after each hidden layer
from keras.layers import Dropout
classifier.add(Dropout(p=0.1)) # p from 0 to 1, 0.1 means dropout 10% of neuron at each hidden layer


# Tuning ANN (Parameter Tuning)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(kernel_initializer='uniform', activation='relu', input_dim=11, output_dim=6))
    classifier.add(Dropout(p=0.25))
    classifier.add(Dense(kernel_initializer='uniform', activation='relu', output_dim=6))
    classifier.add(Dropout(p=0.15))
    classifier.add(Dense(kernel_initializer='uniform', activation='sigmoid', output_dim=1))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32], 
              'epochs': [100, 500],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_