import cv2
import numpy as np


def order_points(pts):
    """Orders points clockwise, starting w top left
    https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(img, points):
    points = order_points(points)
    # Calculate destination rectangle
    x_max, y_max = points.max(axis=0)
    x_min, y_min = points.min(axis=0)
    dst = np.float32([
        [x_max, y_min],
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
    ])
    M = cv2.getPerspectiveTransform(points, dst)
    return cv2.warpPerspective(img, M, img.shape[1::-1])
