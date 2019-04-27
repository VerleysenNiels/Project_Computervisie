import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as optimizers
import keras.losses as losses
import logging


class RandomForestClassifier(object):

    def __init__(self):
        self.classifier = RFC(n_estimators=10, min_samples_leaf=.3)

    def train(self, X, y):
        self.classifier.fit(X, y)

    def eval(self, X, y):
        return self.classifier.score(X, y)

    def infer(self, X):
        return self.classifier.predict(X)


class NeuralNetClassifier(object):

    def __init__(self):
        self.learning_rate = 3e-4
        self.model = Sequential()
        self.model.add(Dense(1024, activation='elu', input_shape=(15, 4489)))
        self.model.add(Dense(256, activation='elu'))
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dense(14, activation='softmax'))
        self.model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=self.learning_rate))

    def train(self, X, y):
        self.model.fit(X, y, verbose=True)

    def eval(self, X, y):
        return self.model.evaluate(X, y)

    def infer(self, X):
        return self.model.predict(X)
