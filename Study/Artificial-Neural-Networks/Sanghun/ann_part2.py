# Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(kernel_initializer='uniform', activation='relu', input_dim=11, output_dim=6))
    classifier.add(Dense(kernel_initializer='uniform', activation='relu', output_dim=6))
    classifier.add(Dense(kernel_initializer='uniform', activation='sigmoid', output_dim=1))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, verbose=1)  # n_jobs=-1 not working on my computer

mean = accuracies.mean()
variance = accuracies.std()
