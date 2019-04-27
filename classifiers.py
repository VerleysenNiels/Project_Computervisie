import cv2
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
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
        self.learning_rate = 1e-3
        self.labels = ["zaal_a", "zaal_b", "zaal_c", "zaal_d", "zaal_e", "zaal_f",
                       "zaal_g", "zaal_h", "zaal_i", "zaal_j", "zaal_k", "zaal_l", "zaal_m", "zaal_n"]
        self.model = self.create_model()

    def train(self, X, y):
        n = len(X)
        idx = np.arange(n)
        np.random.seed(456)
        np.random.shuffle(idx)
        x_shuffle = X[idx, ...]
        y_shuffle = y[idx, ...]
        self.model.fit(x_shuffle, self.transform_output(
            y_shuffle), verbose=True, batch_size=len(X)-1)

    def eval(self, X, y):
        return self.model.evaluate(X, self.transform_output(y))[1]

    def infer(self, X):
        return self.model.predict(X)

    def transform_output(self, y):
        result = []
        for v in y:
            a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            a[self.labels.index(v)] = 1
            result.append(a)
        return result

    def create_model(self):
        model = Sequential()
        model.add(Dense(1024, activation='elu', input_shape=(15360,)))
        model.add(Dense(256, activation='elu'))
        model.add(Dense(256, activation='elu'))
        model.add(Dense(64, activation='elu'))
        model.add(Dense(14, activation='softmax'))
        model.compile(loss=losses.mean_squared_error, optimizer=optimizers.sgd(
            lr=self.learning_rate), metrics=['accuracy'])
        return model
