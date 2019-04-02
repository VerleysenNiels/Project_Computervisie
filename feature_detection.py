import cv2
import numpy as np
import utils


def detect_lines():
    pass


def detect_corners(img, maxCorners=0):
    return cv2.goodFeaturesToTrack(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        maxCorners, .3, 15, useHarrisDetector=True)


if __name__ == "__main__":
    img = cv2.imread('images/single_paintings/still_life/20190217_101231.jpg')
    corners = detect_corners(img)
    img = utils.overlay_points(img, corners)
    utils.imshow(img, resize=True)
