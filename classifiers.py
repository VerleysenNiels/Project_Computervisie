import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifier(object):

    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=10, min_samples_leaf=.3)

    def train(X, y):
        self.classifier.fit(X, y)

    def eval(X, y):
        pass

    def infer(X):
        pass
