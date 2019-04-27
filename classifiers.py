import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
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
